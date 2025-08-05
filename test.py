import argparse
import random
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
import soundfile as sf
from tqdm import tqdm
import librosa
import contextlib
from datetime import datetime

# model
from transformers.models.encodec import EncodecFeatureExtractor
from src.data.dataset import RVTTSDataset
from src.models.rvtts_description import RVTTS
from einops import rearrange
from src.models.face_encoder import Face_Encoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='./imgs')
    parser.add_argument('--text_path', default='./text/text.txt')
    parser.add_argument('--description_path', default='./description/description.txt')

    parser.add_argument('--tokenizer', default='google/flan-t5-base')

    parser.add_argument("--max_sp_len", type=int, default=24, help='max length in sec')
    parser.add_argument("--sr", type=int, default=44100)

    parser.add_argument("--spk_embed_dim", type=int, default=256, help='arcface 512 / contrastive 256')

    parser.add_argument("--no_public", default=False, action='store_true')
    parser.add_argument("--constant_gen", default=False, action='store_true')

    parser.add_argument("--do_sample", default=True)
    parser.add_argument("--top_k", default=30, type=int)
    parser.add_argument("--repetition_penalty", default=1.2, type=float)
    parser.add_argument("--temperature", default=0.9, type=float)

    parser.add_argument("--fe_model_ckpt", default='./pretrained/Face_Encoder.ckpt')

    parser.add_argument("--save_dir", type=str, default='./generated/results')
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/RVTTS')
    parser.add_argument("--checkpoint", type=str, default='./pretrained/RV-TTS.ckpt')

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)

    parser.add_argument("--mode", type=str, default='test', help='train, test, valid')

    parser.add_argument("--fp16", default=False, action='store_true')
    parser.add_argument("--bf16", default=False, action='store_true')

    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument("--torchrun", default=False, action='store_true')
    parser.add_argument("--masterport", type=str, default='1234')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def test(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = args.masterport

    assert args.do_sample, 'Only sampling-based decoding is working'
    assert args.checkpoint is not None

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")
    args.save_dir = os.path.join(args.save_dir) + f"_{dt_string}"
    if args.constant_gen:
        args.save_dir + '_constant'
    if args.no_public:
        args.save_dir + '_no_public'

    if not os.path.exists(args.save_dir) and args.rank == 0:
        os.makedirs(args.save_dir)

    model = RVTTS(spk_embed_dim=args.spk_embed_dim)
    model.model.generation_config.do_sample = args.do_sample
    if args.top_k is not None:
        model.model.generation_config.top_k = args.top_k
    model.model.generation_config.return_dict_in_generate = True
    model.model.generation_config.temperature = args.temperature
    model.model.generation_config.repetition_penalty = args.repetition_penalty

    assert model.config.sampling_rate == args.sr, f'Different sampling rate - config: {model.config.sampling_rate}'

    face_encoder = Face_Encoder('r50')
    checkpoint = torch.load(args.fe_model_ckpt, map_location="cpu", weights_only=True)
    face_encoder.load_state_dict(checkpoint['state_dict'])
    face_encoder.eval()
    face_encoder.cuda()
  
    if args.checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint

    model.cuda()

    if args.constant_gen:
        audio_prep = EncodecFeatureExtractor.from_pretrained("parler-tts/dac_44khZ_8kbps")
        test_generation_constant_sample(model, face_encoder)
        test_generation_constant(model, face_encoder, audio_prep)
    else:
        test_generation(model, face_encoder)
    
def test_generation(model, face_encoder):
    with torch.no_grad():
        model.eval()

        val_data = RVTTSDataset(
            img_path=args.img_path,
            txt_path=args.text_path,
            description_path=args.description_path,
            max_sp_len=args.max_sp_len,
            sr=args.sr,
            no_public=args.no_public,
            constant_gen_sample=False,
            constant_gen_pred=False,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers,
        )

        batch_size = dataloader.batch_size
        samples = int(len(dataloader.dataset))

        embed_description = "The speaker's voice characteristic is "
        global embed_description_feat # 1, S, 768
        with torch.no_grad():
            embed_des_token = val_data.des_tokenizer(embed_description, return_tensors="pt")
            embed_des_feats = model.model.text_encoder(input_ids=embed_des_token.input_ids.cuda())
            embed_description_feat = embed_des_feats[0]

        if args.rank == 0:
            print('Samples Generation')
        for i, batch in enumerate(dataloader):
            if args.rank == 0:
                print(f"******** Generation for Images : {(i + 1) * batch_size} / {samples} ********")
            face_img, f_name = batch

            face_img_tmp = rearrange(face_img.float(), 'B H W C -> B C H W')
            face_img = ((face_img_tmp / 255.) - 0.5) / 0.5
            face_embeds = face_encoder.forward_face(face_img.cuda())
            face_embeds = F.normalize(face_embeds, dim=-1)
            spk_embeds = face_embeds.unsqueeze(1)
            spk_feats = model.face_mapper(spk_embeds)

            num = 0
            for start in tqdm(range(0, len(val_data.des_inp), args.batch_size)):
                description = val_data.des_inp[start:start+args.batch_size]
                description_mask = val_data.des_pad[start:start+args.batch_size]
                prompt = val_data.prt_inp[start:start+args.batch_size]
                prompt_mask = val_data.prt_pad[start:start+args.batch_size]

                des_feats = model.model.text_encoder(input_ids=description.cuda(), attention_mask=description_mask.cuda())
                
                encoder_outputs = torch.cat([embed_description_feat.repeat(prompt.size(0), 1, 1), spk_feats.repeat(prompt.size(0), 1, 1), des_feats[0]], 1)
                encoder_mask = torch.cat([torch.ones([description_mask.size(0), encoder_outputs.size(1) - description_mask.size(1)]), description_mask], 1)
                encoder_outputs = (encoder_outputs, )

                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else torch.autocast(device_type='cuda', dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
                    generation = model.generate(encoder_outputs=encoder_outputs, encoder_attention_mask=encoder_mask.cuda(), prompt_input_ids=prompt.cuda(), prompt_attention_mask=prompt_mask.cuda(), max_new_tokens=2070)
                
                prediction = generation.sequences.float().cpu().numpy()
                audio_len = generation.audios_length

                if args.rank == 0:
                    for (audio, length, fname) in zip(prediction, audio_len, f_name):
                        num += 1
                        save_name = os.path.join(args.save_dir, 'audio', fname + f'_{num}.wav')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        sf.write(save_name, audio[:length], samplerate=args.sr, subtype='PCM_16')

        if args.rank == 0:
            print('#'*10, 'Generation Completed')
        return

def test_generation_constant_sample(model, face_encoder):
    with torch.no_grad():
        model.eval()

        val_data = RVTTSDataset(
            img_path=args.img_path,
            txt_path=args.text_path,
            description_path=args.description_path,
            max_sp_len=args.max_sp_len,
            sr=args.sr,
            no_public=args.no_public,
            constant_gen_sample=True,
            constant_gen_pred=False,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers,
        )

        batch_size = dataloader.batch_size
        samples = int(len(dataloader.dataset))

        embed_description = "The speaker's voice characteristic is "
        global embed_description_feat # 1, S, 768
        with torch.no_grad():
            embed_des_token = val_data.des_tokenizer(embed_description, return_tensors="pt")
            embed_des_feats = model.model.text_encoder(input_ids=embed_des_token.input_ids.cuda())
            embed_description_feat = embed_des_feats[0]

        if args.rank == 0:
            print('Stage 1: Voice Samples Generation')
        for i, batch in enumerate(dataloader):
            if args.rank == 0:
                print(f"******** Generation for Images : {(i + 1) * batch_size} / {samples} ********")
            face_img, f_name = batch

            face_img_tmp = rearrange(face_img.float(), 'B H W C -> B C H W')
            face_img = ((face_img_tmp / 255.) - 0.5) / 0.5
            face_embeds = face_encoder.forward_face(face_img.cuda())
            face_embeds = F.normalize(face_embeds, dim=-1)
            spk_embeds = face_embeds.unsqueeze(1)
            spk_feats = model.face_mapper(spk_embeds)

            for _, (description, description_mask, prompt, prompt_mask) in enumerate(zip(val_data.des_inp, val_data.des_pad, val_data.prt_inp, val_data.prt_pad)):
                description, description_mask, prompt, prompt_mask =  description.unsqueeze(0), description_mask.unsqueeze(0), prompt.unsqueeze(0), prompt_mask.unsqueeze(0)
                des_feats = model.model.text_encoder(input_ids=description.cuda(), attention_mask=description_mask.cuda())
                
                encoder_outputs = torch.cat([embed_description_feat.repeat(prompt.size(0), 1, 1), spk_feats.repeat(prompt.size(0), 1, 1), des_feats[0]], 1)
                encoder_mask = torch.cat([torch.ones([description_mask.size(0), encoder_outputs.size(1) - description_mask.size(1)]), description_mask], 1)
                encoder_outputs = (encoder_outputs, )

                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else torch.autocast(device_type='cuda', dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
                    generation = model.generate(encoder_outputs=encoder_outputs, encoder_attention_mask=encoder_mask.cuda(), prompt_input_ids=prompt.cuda(), prompt_attention_mask=prompt_mask.cuda(), max_new_tokens=2070)
                
                prediction = generation.sequences.float().cpu().numpy()
                audio_len = generation.audios_length

            if args.rank == 0:
                for (audio, length, fname) in zip(prediction, audio_len, f_name):
                    save_name = os.path.join(args.save_dir, 'voice_sample', fname + '.wav')
                    if not os.path.exists(os.path.dirname(save_name)):
                        os.makedirs(os.path.dirname(save_name))
                    sf.write(save_name, audio[:length], samplerate=args.sr, subtype='PCM_16')

        if args.rank == 0:
            print('#'*10, 'Voice Sample Generation Completed')

def test_generation_constant(model, face_encoder, audio_prep):
    with torch.no_grad():
        model.eval()

        val_data = RVTTSDataset(
            img_path=args.img_path,
            txt_path=args.text_path,
            description_path=args.description_path,
            max_sp_len=args.max_sp_len,
            sr=args.sr,
            no_public=args.no_public,
            constant_gen_sample=False,
            constant_gen_pred=True,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers,
        )

        batch_size = dataloader.batch_size
        samples = int(len(dataloader.dataset))

        embed_description = "The speaker's voice characteristic is "
        global embed_description_feat # 1, S, 768
        with torch.no_grad():
            embed_des_token = val_data.des_tokenizer(embed_description, return_tensors="pt")
            embed_des_feats = model.model.text_encoder(input_ids=embed_des_token.input_ids.cuda())
            embed_description_feat = embed_des_feats[0]

        if args.rank == 0:
            print('Stage 2: Samples Generation based on Generated Voice Samples')
        for i, batch in enumerate(dataloader):
            if args.rank == 0:
                print(f"******** Generation for Images : {(i + 1) * batch_size} / {samples} ********")
            face_img, f_name = batch
            
            speaker = f_name[0]
            cont_audio, _ = librosa.load(os.path.join(args.save_dir, 'voice_sample', f'{speaker}.wav'), sr=args.sr)
            cont_audio = torch.FloatTensor(cont_audio)
            input_values = audio_prep(cont_audio, sampling_rate=args.sr, return_tensors="pt")["input_values"]

            face_img_tmp = rearrange(face_img.float(), 'B H W C -> B C H W')
            face_img = ((face_img_tmp / 255.) - 0.5) / 0.5
            face_embeds = face_encoder.forward_face(face_img.cuda())
            face_embeds = F.normalize(face_embeds, dim=-1)
            spk_embeds = face_embeds.unsqueeze(1)
            spk_feats = model.face_mapper(spk_embeds)

            num = 0
            for start in tqdm(range(0, len(val_data.des_inp), args.batch_size)):
                description = val_data.des_inp[start:start+args.batch_size]
                description_mask = val_data.des_pad[start:start+args.batch_size]
                prompt = val_data.prt_inp[start:start+args.batch_size]
                prompt_mask = val_data.prt_pad[start:start+args.batch_size]

                des_feats = model.model.text_encoder(input_ids=description.cuda(), attention_mask=description_mask.cuda())
                
                encoder_outputs = torch.cat([embed_description_feat.repeat(prompt.size(0), 1, 1), spk_feats.repeat(prompt.size(0), 1, 1), des_feats[0]], 1)
                encoder_mask = torch.cat([torch.ones([description_mask.size(0), encoder_outputs.size(1) - description_mask.size(1)]), description_mask], 1)
                encoder_outputs = (encoder_outputs, )

                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else torch.autocast(device_type='cuda', dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
                    generation = model.generate(encoder_outputs=encoder_outputs, encoder_attention_mask=encoder_mask.cuda(), prompt_input_ids=prompt.cuda(), prompt_attention_mask=prompt_mask.cuda(), max_new_tokens=2070, input_values=input_values.repeat(prompt.size(0), 1, 1).cuda())
                
                prediction = generation.sequences.float().cpu().numpy()
                audio_len = generation.audios_length
                f_names = [f_name[0]] * prompt.size(0)

                if args.rank == 0:
                    for (audio, length, fname) in zip(prediction, audio_len, f_names):
                        num += 1
                        save_name = os.path.join(args.save_dir, 'audio', fname + f'_{num}.wav')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        sf.write(save_name, audio[cont_audio.size(0):length], samplerate=args.sr, subtype='PCM_16')

        if args.rank == 0:
            print('#'*10, 'Generation Completed')
        return

if __name__ == "__main__":
    args = parse_args()
    test(args)