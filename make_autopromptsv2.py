import argparse
import torch
import os
from vqa_dataloader import make_dataloader
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from OFA.transformers.src.transformers.models.ofa.generate import sequence_generator
from collections import defaultdict
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
#parser.add_argument('--cls_names', type=str, default='polyp',required=True)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--dataset', type=str, default='dfuc', required=True)
parser.add_argument('--bert_type', type=str, default='pubmed')
parser.add_argument('--ofa_type', type=str, default='base')
parser.add_argument('--mode', type=str, default='hybrid', help='if both will generate lama and vqa and hybird')
parser.add_argument('--cls_names', action='append', required=True)
parser.add_argument('--real_cls_names', action='append', required=True)
parser.add_argument('--vqa_names', action='append', required=True)

args = parser.parse_args()

bert_map = {
    'pubmed': "./BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    'bert': None,
}
bert_path = bert_map[args.bert_type]

ofa_map ={
    'base': 'ofa-base/',
}
ofa_path = ofa_map[args.ofa_type]

DATASETMAP = {
    'dfuc': {'data_path': 'DATA/DFUC/images/dfuc2020_val', 'anno_path': 'DATA/DFUC/annotations/dfuc2020_val.json'},
    'isbi': {'data_path':'DATA/ISBI2016/images/test', 'anno_path':'DATA/ISBI2016/annotations/test.json'},
    'cvc300': {'data_path':'DATA/POLYP/val/CVC-300/images', 'anno_path':'DATA/POLYP/annotations/CVC-300_val.json'},
    'colondb': {'data_path':'DATA/POLYP/val/CVC-ColonDB/images', 'anno_path':'DATA/POLYP/annotations/CVC-ColonDB_val.json'},
    'clinicdb': {'data_path':'DATA/POLYP/val/CVC-ClinicDB/images', 'anno_path':'DATA/POLYP/annotations/CVC-ClinicDB_val.json'},
    'kvasir': {'data_path':'DATA/POLYP/val/Kvasir/images', 'anno_path':'DATA/POLYP/annotations/Kvasir_val.json'},
    'warwick': {'data_path': 'DATA/WarwickQU/images/test', 'anno_path': 'DATA/WarwickQU/annotations/test.json'},
    'bccd': {'data_path': 'DATA/BCCD/test', 'anno_path': 'DATA/BCCD/annotations/test.json'},
    'cpm17': {'data_path': 'DATA/Histopathy/cpm17/images/test', 'anno_path': 'DATA/Histopathy/cpm17/annotations/test.json'}
}

resolution = 384
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
patch_resize_transform = transforms.Compose([
    lambda image: image.convert('RGB'),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std)
    ])

def build_model(bert_path, ofa_path,mode='hybrid'):
    '''
    if mode == hybrid, will generate lama and vqa hybrid prompt
    if mode == all, will generate lama prompt for whole data set
    '''
    if mode == 'hybrid' or mode == 'location':
        tokenizer_lama = AutoTokenizer.from_pretrained(bert_path)
        model_lama = AutoModelForMaskedLM.from_pretrained(bert_path).to('cuda')
        model_lama.eval()

        tokenizer_vqa = OFATokenizer.from_pretrained(ofa_path, use_cache=True)
        model_vqa = OFAModel.from_pretrained(ofa_path, use_cache=False)
        model_vqa.eval()

        return model_lama,tokenizer_lama, model_vqa, tokenizer_vqa

    elif mode == 'lama':
        tokenizer_lama = AutoTokenizer.from_pretrained(bert_path)
        model_lama = AutoModelForMaskedLM.from_pretrained(bert_path).to('cuda')
        model_lama.eval()

        return model_lama, tokenizer_lama, None, None

def masked_prompt(cls_names, model, tokenizer, mode='hybrid', topk=3):
    '''
    cls_names are the name of each class as a list
    return a prompt info dict:
                {'cls_1': {'location': top1, top2, top3}}
    '''
    res = defaultdict(dict)
    for cls_name in cls_names:
        if mode == 'hybrid':
            questions_dict = {
                #'location': f'[CLS] The location of {cls_name} is at [MASK] . [SEP]', #num of mask?
                'location': f'[CLS] Only [MASK] cells have a {cls_name}. [SEP]'
                # 'modality': in [mask] check, we will find polyp
                # 'color': f'[CLS] The typical color of {cls_name} is [MASK] . [SEP]',
                #'shape': f'[CLS] The shape of {cls_name} is [MASK] . [SEP]',
                #'def': f'{cls_name} is a  . [SEP]',
            }
        elif mode == 'lama':
            # questions_dict = {
            #     'location': f'[CLS] The location of {cls_name} is at [MASK] . [SEP]', #num of mask?
            #     'color': f'[CLS] In a fundus photography, the {cls_name} is in [MASK] color . [SEP]',
            #     'shape': f'[CLS] In a fundus photography, the {cls_name} is [MASK] shape . [SEP]',
            #     #'def': f'{cls_name} is a  . [SEP]',
            # }
            questions_dict = {
            #'location': f'[CLS] Only [MASK] cells have a {cls_name}. [SEP]', #num of mask?
            # 'location': f'[CLS] The {cls_name} normally appears at or near the [MASK] of a cell. [SEP]',
            # 'color': f'[CLS] When a cell is histologically stained, the {cls_name} are in [MASK] color. [SEP]',
            # 'shape': f'[CLS] Mostly the shape of {cls_name} is [MASK]. [SEP]',
            'location': f'[CLS] The location of {cls_name} is at [MASK]. [SEP]',
            'color': f'[CLS] The typical color of {cls_name} is [MASK]. [SEP]',
            'shape': f'[CLS] The typical shape of {cls_name} is [MASK]. [SEP]',
            #'def': f'{cls_name} is a  . [SEP]',
        }

        elif mode == 'location':
            return None


        res[cls_name] = defaultdict(list)
        for k, v in questions_dict.items():
            # import pdb; pdb.set_trace()
            predicted_tokens = []
            input_ids_translated = tokenizer(
                v,
                return_tensors = 'pt'
                ).input_ids.to('cuda')
            tokenized_text = tokenizer.tokenize(v)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            # Create the segments tensors.
            segments_ids = [0] * len(tokenized_text)
            
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            segments_tensors = torch.tensor([segments_ids]).to('cuda')

            masked_index = tokenized_text.index('[MASK]')
            with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensors)
            
            _, predicted_index = torch.topk(predictions[0][0][masked_index], topk)#.item()
            predicted_index = predicted_index.detach().cpu().numpy()
            #print(predicted_index)
            for idx in predicted_index:
                predicted_tokens.append(tokenizer.convert_ids_to_tokens([idx])[0])
            #print(predicted_tokens)
            temp_str = v.strip().split(' ')
            for i in range(topk):
                #print(predicted_tokens[i])
                #temp_str2 = [predicted_tokens[i] if s == '[MASK]' else s for s in temp_str]
                #print(temp_str2)
                #pred_translated = " ".join(temp_str2.copy())
                #print(pred_translated)
                res[cls_name][k].append(predicted_tokens[i])
            #print(pred_translated)
    return res

def create_prompt(cls_names, imgs, tokenizer, model, vqa_names, real_names, lama_knowledge, mode='hybrid',topk=3):
    cls_nums = len(cls_names)
    
    if mode == 'hybrid':
        prompt_dict = {
        'color': [f'What is the color of these {vqa_names[i]}?' for i in range(cls_nums)],
        'shape': [f'What is the shape of these {vqa_names[j]}?' for j in range(cls_nums)],
                    }
    elif mode == 'location':
        prompt_dict = {
        'location': [f'Where is this {vqa_names[j]} located on?' for j in range(cls_nums)],
        }

    #caption = {'prefix':'', 'name':'', 'suffix':''}
    captions, caption = dict(), dict()
    #loc_captions = dict()
    ans_dict = dict()

    for img in imgs:
        #import pdb; pdb.set_trace()

        # generate prompt
        img = img.unsqueeze(0)
        cls_dict = {}
        for j in range(topk):
            caption = {'caption':[],
            'prefix': [],
            'suffix': [],
            'name': [],
            }
            
            loc_caption = {'location': []}
            for i in range(cls_nums):
                for k, v in prompt_dict.items():
                    #import pdb; pdb.set_trace()
                    if j >= 1:
                        # vqa generated content is the same
                        continue
                    txt = v[i]
                    inputs = tokenizer([txt], return_tensors="pt").input_ids
                    gen = model.generate(inputs, patch_images=img, num_beams=10, no_repeat_ngram_size=3)
                    gen = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
                    ans_dict[k] = gen.strip()#[:-1] #no period sign
                if mode == 'location':
                    #location = ans_dict['location']
                    loc_caption['location'].append(ans_dict['location'])
                    continue
                if j < 1:
                    color, shape = ans_dict['color'], ans_dict['shape']
                    cls_dict[i] = dict()
                    cls_dict[i]['color'] = color
                    cls_dict[i]['shape'] = shape
                lama_info = lama_knowledge[cls_names[i]]
                #topk_num = len(lama_info)
                # color, shape, location = ans_dict['color'], ans_dict['shape'], lama_info['location'][j]
                color, shape, location = cls_dict[i]['color'], cls_dict[i]['shape'], lama_info['location'][j]
                name = f'{real_names[i]}'#f'{color} {shape} {real_names[i]}'
                prefix, suffix = f'{color} ', f' in {location} cells'#f' {location}'
                caption['prefix'] += [prefix]
                caption['name'] += [name]
                caption['suffix'] += [suffix]
                caption['caption'] += [prefix + name + suffix]
                caption['color'] = color
                caption['shape'] = shape
                caption['cls'] = real_names[i] 
            
            if mode == 'location':
                captions[j] = loc_caption
            elif mode == 'hybrid':
                caption['caption'] = '. '.join(caption['caption'])
                captions[j] = caption
    return captions

def gen_prompt(bert_path, ofa_path, args):
    model_lama, tokenizer_lama, model_vqa, tokenizer_vqa = build_model(bert_path, ofa_path, args.mode)
    cls_nums = len(args.cls_names)
    lama_knowledge = masked_prompt(args.cls_names, model_lama, tokenizer_lama, args.mode, args.topk)
    root, annoFile = DATASETMAP[args.dataset]['data_path'], DATASETMAP[args.dataset]['anno_path']
    data_loader = make_dataloader(root, annoFile, patch_resize_transform)
    _iterator = tqdm(data_loader)
    # prompt_top1 = {'prompts': []}
    # prompt_top2 = {'prompts': []}
    # prompt_top3 = {'prompts': []}
    prompt_top1, prompt_top2, prompt_top3 = dict(), dict(), dict()
    #TODO automize this shit
    #import pdb; pdb.set_trace()
    for i, batch in enumerate(_iterator):
        images, targets, paths, *_ = batch
        if args.mode == 'location':
            prompts_dict = create_prompt(args.cls_names, images, tokenizer_vqa, model_vqa,
                                         args.vqa_names, args.real_cls_names,
                                        lama_knowledge, args.mode, args.topk)
            for i, path in enumerate(paths):
                prompt_top1[path] = prompts_dict[0]
                prompt_top2[path] = prompts_dict[1]
                prompt_top3[path] = prompts_dict[2]
            # for j in range(args.topk):
            #     locations = []
            #     for i, cls_name in enumerate(args.cls_names):
            #         location = lama_knowledge[cls_name]['color'][j]
            #         locations.append(location)
            # for i, path in enumerate(paths): #paths batchsize=1
            #     prompt_top1[path] = locations[0]
            #     prompt_top2[path] = locations[1]
            #     prompt_top3[path] = locations[2]
                    
        elif args.mode == 'lama':
            # import pdb; pdb.set_trace()
            captions, caption = [], dict()

            for j in range(args.topk):
                caption = {'caption':[],
                            'prefix': [],
                            'suffix': [],
                            'name': [],
                            }
                for i, cls_name in enumerate(args.cls_names):
                    color, shape, location = lama_knowledge[cls_name]['color'][j],\
                                             lama_knowledge[cls_name]['shape'][j], \
                                             lama_knowledge[cls_name]['location'][j]
                                             
                    #name = f'{color} {shape} {args.real_cls_names[i]}'
                    name = args.real_cls_names[i]
                    prefix, suffix = f'{color} color, {shape} shape ', f' in {location}'
                    caption['prefix'] += [prefix]
                    caption['name'] += [name]
                    caption['suffix'] += [suffix]
                    caption['caption'] += [prefix + name + suffix]
                caption['caption'] = '. '.join(caption['caption'])
                captions.append(caption)
            #import pdb; pdb.set_trace()
            for i, path in enumerate(paths): #paths batchsize=1
                prompt_top1[path] = captions[0]
                prompt_top2[path] = captions[1]
                prompt_top3[path] = captions[2]

        elif args.mode == 'hybrid':
            prompts_dict = create_prompt(args.cls_names, images, tokenizer_vqa, model_vqa,
                                         args.vqa_names, args.real_cls_names,
                                        lama_knowledge, args.mode, args.topk)
            # prompt_top1['prompts'] += [prompts_dict[0]]
            # prompt_top2['prompts'] += [prompts_dict[1]]
            # prompt_top3['prompts'] += [prompts_dict[2]]
            #import pdb;pdb.set_trace()
            for i, path in enumerate(paths):
                prompt_top1[path] = prompts_dict[0]
                prompt_top2[path] = prompts_dict[1]
                prompt_top3[path] = prompts_dict[2]
    
    # prompt_json1 = json.dumps(prompt_top1)
    # prompt_json2 = json.dumps(prompt_top2)
    # prompt_json3 = json.dumps(prompt_top3)

    with open(f'autoprompt_json/{args.mode}_{args.dataset}_path_prompt_top1.json', 'w') as f1:
        json.dump(prompt_top1, f1)
    
    with open(f'autoprompt_json/{args.mode}_{args.dataset}_path_prompt_top2.json', 'w') as f2:
        json.dump(prompt_top2, f2)

    with open(f'autoprompt_json/{args.mode}_{args.dataset}_path_prompt_top3.json', 'w') as f3:
        json.dump(prompt_top3, f3)

gen_prompt(bert_path, ofa_path, args)