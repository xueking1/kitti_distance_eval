import time
import argparse
import kitti_common as kitti
from eval import get_official_eval_result_by_distance, get_official_eval_result, get_coco_eval_result
import pickle
import numpy as np
import datetime
import logging
def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
    
def printer_decorator(func):
    def print_results(result):
        for r in result:
            if type(r)==str:
                print(r)
            elif type(r)==list:
                print_results(r)
            elif type(r)==dict:
                for kw in r:
                    print(kw,':',r[kw])

    def printer(*args, **kwargs):
        default_res, distance_res = func(*args, **kwargs)
        print("--------------- RESULTS by DIFFICULTY ---------------")
        print_results(default_res)
        print("---------------  RESULTS by DISTANCE  ---------------")
        print_results(distance_res)

    return printer

@printer_decorator
def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1):
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return [get_official_eval_result(gt_annos, dt_annos, current_class), get_official_eval_result_by_distance(gt_annos, dt_annos, current_class)]


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info
        

if __name__ == '__main__':
    # fire.Fire()
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--save_path', type=str, default="/path/to/your_result_folder",
                        help='specify the config for training')
    parser.add_argument('--gt_split_file', type=str, default="/path/to/val.txt",
                        help='specify the config for training')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--sampled_interval', type=int, default=1, help='sampled interval for GT sequences')
    args = parser.parse_args()
    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))
    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]['annos']
        # cur_info['frame_id'] = gt_infos[idx]['annos']
        cur_info = drop_info_with_name(cur_info, name='DontCare')  # discard DontCare
        gt_names = cur_info['name']
        cur_info['name'] = np.array(['Car' if gt_names[i] == 'Van' else gt_names[i] for i in range(len(gt_names))])
        cur_info['frame_id'] = gt_infos[idx]['point_cloud']['lidar_idx']
        gt_infos_dst.append(cur_info)

    gt_annos = gt_infos_dst
    dt_annos = pred_infos
    # dt_annos = kitti.get_label_annos(det_path)
    # val_image_ids = _read_imageset_file(gt_split_file)
    # gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
    # print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer

    log_file = args.save_path + '/'+ ('distance_log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = create_logger(log_file)

    # log to file
    logger.info('**********************Start logging**********************')
    _, results = get_official_eval_result_by_distance(gt_annos, dt_annos,0)
    for key in results:
        print( key +': '+ str(results[key]))
        logger.info(key +': '+ str(results[key]))
    # print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer