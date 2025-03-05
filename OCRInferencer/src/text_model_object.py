from pathlib import Path

import mmcv
from mmocr.apis import MMOCRInferencer
# from mmocr.datasets.pipelines.crop import crop_img

'''
Usable with Docker image mmocr:ver2.4
'''

class TextModel():
    def __init__(self, det: str, recog: str, det_thresh: float=0, rec_thresh: float=0, home_path='/home/src'):
        self.det = det if isinstance(det, str) else 'DB_r50'
        self.recog = recog if isinstance(recog, str) else 'ABINet'
        self.det_thresh = det_thresh if (isinstance(det_thresh, float) and 0 <= det_thresh <= 1) else 0 
        self.rec_thresh = rec_thresh if (isinstance(rec_thresh, float) and 0 <= rec_thresh <= 1) else 0
        self.text_engine = None
        self.det_model = None
        self.recog_model = None
        self.inferencer = None
        self.home_path = home_path if isinstance(home_path, str) else None
        self.textdet_models = {
            'DB_r18': {
                'config': 'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                # 'dbnet/dbnet_resnet50-oclip_1200e_icdar2015.py',
                'dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                # 'checkpoints/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth'
                'dbnetpp_finetuned.pth'
            },
            'DB_r50_oclip':{
                'config':'dbnet/dbnet_resnet50-oclip_1200e_icdar2015.py',
                'ckpt':'dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth',
            },
            'DBpp_finetune': {
                'config': 'dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015.py',
                'ckpt': 'dbnetpp_finetuned.pth'
            },
            'DRRG': {
                'config': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt': 'drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth'
            },
            'FCE_IC15': {
                'config': 'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt': 'fcenet_r50_fpn_1500e_icdar2015-d435c061.pth'
            },
            'FCE_CTW_DCNv2': {
                'config': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt': 'fcenet_r50dcnv2_fpn_1500e_ctw1500-05d740bb.pth'
            },
            'MaskRCNN_CTW': {
                'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt': 'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config': 'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config': 'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt': 'psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config': 'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt': 'psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config': 'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt': 'textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
            }
        }
        self.textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt': 'nrtr_r31_academic_20210406-954db95e.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt': 'nrtr_r31_1by8_1by4_academic_20210406-ce16e7cc.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner_r31_academic-5f05874f.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'crnn_tps_academic_dataset_20210510-d221a905.pth'
            },
            'ABINet': {
                'config': 'abinet/abinet_20e_st-an_mj.py',
                'ckpt': 'abinet_20e_st-an_mj_20221005_012617-ead8c139.pth'
            },
            'ABINetVision': {
                'config': 'abinet/abinet-vision_20e_st-an_mj.py',
                'ckpt': 'abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth'
            }
        }

    def start_engine(self) -> None:
        dir_path = self.home_path

        det_config = dir_path + '/src/configs/textdet/' + self.textdet_models[self.det]['config']
        det_ckpt = dir_path + '/assets/' + self.textdet_models[self.det]['ckpt']
        
        recog_config = dir_path + '/src/configs/textrecog/' + self.textrecog_models[self.recog]['config']
        recog_ckpt = dir_path + '/assets/' + self.textrecog_models[self.recog]['ckpt']

        print(det_config, recog_config)

        self.inferencer = MMOCRInferencer(
            det=det_config,
            det_weights=det_ckpt,
            rec=recog_config,
            rec_weights=recog_ckpt
        )
        
    
    # End2end ocr inference pipeline
    def det_and_recog_inference(self, input):
        results = []
        dropped = 0
        det_result = self.inferencer(input)
        tmp_result = det_result['predictions'][0]
        print(det_result['predictions'])
        for poly, text, d_score, r_score in zip(tmp_result['det_polygons'], tmp_result['rec_texts'], tmp_result['det_scores'], tmp_result['rec_scores']):
            box_res = {
                    'bbox': None,
                    'attributes': {},
                    'score': None
            }
            l = min(poly[0::2])
            t = min(poly[1::2])
            w = max(poly[0::2]) - l
            h = max(poly[1::2]) - t
            # box_res['bbox'] = [l,t,w,h]
            box_res['x'] = l
            box_res['y'] = t
            box_res['w'] = w
            box_res['h'] = h
            box_res['Text'] = text
            box_res['d_score'] = r_score
            box_res['r_score'] = d_score 
            if box_res['d_score'] > self.det_thresh and box_res['r_score'] > self.rec_thresh:
                results.append(box_res)
            else:
                dropped += 1
        print("Items dropped for this frame: {}".format(dropped))
        return results