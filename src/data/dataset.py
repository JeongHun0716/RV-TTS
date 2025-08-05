import os
import random
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import onnxruntime as ort
import glob 

class RVTTSDataset(Dataset):
    def __init__(
            self,
            img_path='./imgs',
            txt_path='./text/text.txt',
            description_path='./description/description.txt',
            tokenizer='google/flan-t5-base',
            mode='test', 
            max_sp_len=24,  #24 sec
            sr=44100,
            no_public=False,
            constant_gen_sample=False,
            constant_gen_pred=False,
            ):
        
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.sr = sr
        self.no_public = no_public
        self.constant_gen_sample = constant_gen_sample
        self.constant_gen_pred = constant_gen_pred

        _default_session_options = ort.capi._pybind_state.get_default_session_options()
        def get_default_session_options_new():
            _default_session_options.inter_op_num_threads = 1
            _default_session_options.intra_op_num_threads = 1
            return _default_session_options
        ort.capi._pybind_state.get_default_session_options = get_default_session_options_new
        self.app = FaceAnalysis(allowed_modules=['detection'], providers=['CPUExecutionProvider']) # enable detection model only
        self.app.prepare(ctx_id=0, det_size=(224, 224))

        self.constant_prompt = "voice sample."

        self.public = [
            'The recording is made in a public speaking setting.',
            'The audio is recorded in a public speech environment.',
            'The capture takes place in a public speaking context.',
            'The recording occurs in a public address environment.',
            'The audio is captured during a public speech event.',
            'The recording is done in a public speaking atmosphere.',
            'The audio is taken in a public speech setting.',
            'The capture happens in a public speaking environment.',
            'The recording is conducted in a public address context.',
            'The audio is recorded in a public speaking scenario.',
            ]

        self.basic_description = "A person speaks with a monotone voice at a moderate pace. The recording is almost noiseless and the speaker's voice comes across as very close and clear"

        self.imgs = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
                
        if self.constant_gen_sample:
            descs = [self.basic_description]
            txts = [self.constant_prompt]
        else:
            descs = []
            with open(description_path, 'r') as txt:
                lines = txt.readlines()
            for l in lines:
                if not self.no_public:
                    additional_des = random.sample(self.public, 1)[0]
                    l = l.strip() + ' ' + additional_des
                else:
                    l = l.strip()
                descs.append(l)

            txts = []
            with open(txt_path, 'r') as txt:
                lines = txt.readlines()
            for l in lines:
                if self.constant_gen_pred:
                    l = self.constant_prompt + ' ' + l.strip().lower()
                else:
                    l = l.strip()
                txts.append(l.strip().lower())
            
        assert len(txts) == len(descs), "Input text and description should be the same number of lines"

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_side="left")
        self.des_tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_side="right")
        self.max_sp_len = max_sp_len

        prompt = self.tokenizer(txts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        self.prt_inp = prompt.input_ids
        self.prt_pad = prompt.attention_mask

        description = self.des_tokenizer(descs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        self.des_inp = description.input_ids
        self.des_pad = description.attention_mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]

        image = cv2.imread(img)
        image = image[:, :, ::-1]   # BGR -> RGB

        faces = self.app.get(image)
        if len(faces) > 0:
            aligned_im = face_align.norm_crop(image, faces[0]['kps'], 112) #aligned image
        else:   # center crop
            print('No face detected from', img)
            x_c = image.shape[0] // 2
            y_c = image.shape[1] // 2
            aligned_im = image[x_c - 56:x_c + 56,y_c - 56:y_c + 56]
    
        image = aligned_im
        return image, os.path.splitext(os.path.basename(img))[0]