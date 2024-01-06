# PyTorch model testing script

import re
import cv2
import torch
import click
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from pathlib import Path

from dataset import RailSem19Dataset

FOURCC = cv2.VideoWriter_fourcc(*"vp09")

def get_vis_augmentation():
    transform = [
        A.LongestMaxSize(720, cv2.INTER_AREA),
        A.PadIfNeeded(720, 720, cv2.BORDER_CONSTANT),
    ]
    return A.Compose(transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    transform = []
    if preprocessing_fn:
        transform.append(A.Lambda(image=preprocessing_fn))
    transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
    return A.Compose(transform)

def find_intersection(pred_mask, dataset, threshold, base_class='rail-raised', intersect_class='terrain'):
    ch = dataset.classes[base_class]['dim']
    base_mask = pred_mask[:,:, ch].squeeze()
    base_mask = np.where(base_mask >= threshold, 255, 0).astype('uint8')

    x0, y0 = 0, 0
    x1, y1 = base_mask.shape[1], int(base_mask.shape[0] * .4)
    base_mask[y0:y1, x0:x1] = 0

    x0, y0 = 0, int(base_mask.shape[0] * .9)
    x1, y1 = base_mask.shape[1], base_mask.shape[0]
    base_mask[y0:y1, x0:x1] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return np.array([])

    contours_sorted = sorted(contours, key=cv2.contourArea)

    blank = np.zeros(pred_mask.shape[:2], dtype='uint8')
    h, w = blank.shape[:2]
    rx0, rx1 = blank.shape[1] // 6, blank.shape[1] // 2
    deltas = ((rx0, rx1), (blank.shape[1] // 2, blank.shape[1] // 6))

    for cnt, xd in zip(contours_sorted[-2:], deltas):
        vect = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x,y = tuple(vect.flatten())
        lefty = int(((xd[0]-x)*vy/vx) + y)
        righty = int(((w-xd[1]-x)*vy/vx) + y)
        cv2.line(blank,(w-xd[1], righty),(xd[0], lefty), 1, 10)

    ch = dataset.classes[intersect_class]['dim']
    intersect_mask = pred_mask[:,:, ch].squeeze()
    blank = np.where(intersect_mask >= threshold, blank+1, blank)

    return blank > 1

def process_frame(image, model, dataset, display_classes, device, threshold, show_obstacles=False):
    orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aug_image = dataset.augmentation(image=orig_image)['image']
    pp_image = dataset.preprocessing(image=aug_image)['image']

    x_tensor = torch.from_numpy(pp_image).to(device).unsqueeze(0)
    pred_mask = model.predict(x_tensor)
    pred_mask = pred_mask.cpu().numpy().squeeze().transpose(1, 2, 0)

    output_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
    for label in display_classes:
        ch = dataset.classes[label]['dim']
        color = dataset.classes[label]['color']
        m = pred_mask[:,:, ch].squeeze()
        m_idx = m >= threshold
        output_image[m_idx] = output_image[m_idx]*0.6 + np.array(color)*0.4

    if show_obstacles:
        intersect_mask = find_intersection(pred_mask, dataset, threshold)
        if intersect_mask.size > 0:
            (y,x), r = cv2.minEnclosingCircle(np.argwhere(intersect_mask))
            cv2.circle(output_image, (int(x), int(y)), int(r), (0,0,255), 2)

    return aug_image, output_image

@click.command
@click.option('-i', '--image', 'image_path', 
              type=click.Path(exists=True, dir_okay=False), default='./input/sample.png',
              help='Image file path')
@click.option('-v', '--video', 'video_path', 
              type=click.Path(exists=True, dir_okay=False), default=None,
              help='Video file path')
@click.option('-d', '--data_dir', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True), default='./data/rs19_val/',
              help='Location of the RS19 dataset')
@click.option('-o', '--output', 
              type=click.Path(exists=False, file_okay=True, writable=True),
              help='Save result to file')
@click.option('-t', '--threshold', type=click.FloatRange(0.1, 1.0), default=0.8,
              help='Confidence threshold')
@click.option('-f', '--filter', 'class_filter',
              help='Class filter (list of regex expressions separated by comma)')
@click.option('--use_cpu', is_flag=True, help='Run inference on CPU')
@click.option('--labels', 'show_labels', is_flag=True, help='Display RailSem19 dataset labels and quit')
@click.option('-x', '--obstacle', 'show_obstacles', is_flag=True, 
              help='Find and highlight obstacles on railway (images only)')
def main(image_path, video_path, data_dir, output, threshold, class_filter, use_cpu, show_labels, show_obstacles):
    """ Model test """

    device = 'cuda' if not use_cpu else 'cpu'
    images_dir = str(Path(data_dir).joinpath('jpgs', 'rs19_val'))
    masks_dir = str(Path(data_dir).joinpath('uint8', 'rs19_val'))
    config_json_path = str(Path(data_dir).joinpath('rs19-config.json'))
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet101', 'imagenet')

    dataset = RailSem19Dataset(
        images_dir, 
        masks_dir, 
        config_json_path,
        0,
        augmentation = get_vis_augmentation(),
        preprocessing = get_preprocessing(preprocessing_fn)
    )
    if show_labels:
        print(f'RS19 labels:\n---')
        for label, item in dataset.classes.items():
            print(f'{item["id"]}: {label}')
        return
    
    best_model = torch.load('./weights/rs19_deeplabv3plus.pth')

    if not class_filter:
        display_classes = list(dataset.classes.keys())
    else:
        patterns = [s.strip() for s in class_filter.split(',')]
        display_classes = []
        for cls in dataset.classes:
            for p in patterns:
                if re.search(p, cls):
                    display_classes.append(cls)
                    break

    legend = np.full(((len(display_classes) + int(show_obstacles))*14 + 6, 200, 3), (255,255,255), dtype=np.uint8)
    xt, yt = 5, 5
    for label in display_classes:
        cls_id = dataset.classes[label]['id']
        color = dataset.classes[label]['color']
        cv2.rectangle(legend, (xt, yt), (xt+10, yt+10), color, -1)
        cv2.putText(legend, f'{cls_id}: {label}',(xt+13, yt+9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        yt += 14

    if not video_path:
        if (image := cv2.imread(image_path)) is None:
            raise Exception(f'Cannot open image file {image_path}')

        if show_obstacles:
            cv2.rectangle(legend, (xt, yt), (xt+10, yt+10), (0,0,255), -1)
            cv2.putText(legend, 'obstacles',(xt+13, yt+9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        aug_image, output_image = process_frame(image, best_model, dataset, display_classes, device, threshold, show_obstacles)

        x, y, w, h = 0, 0, legend.shape[1], legend.shape[0]
        output_image[y:y+h, x:x+w] = cv2.addWeighted(output_image[y:y+h, x:x+w], 0.2, legend, 0.8, 0.0)

        aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        stacked_image = np.hstack((aug_image, output_image))

        cv2.imshow(f'{image_path}', stacked_image)
        cv2.waitKey(0)

        if output:
            cv2.imwrite(output, stacked_image)
            print(f'Image saved to {output}')
    else:
        stream = cv2.VideoCapture(video_path)
        output_stream = None

        while (buf := stream.read()) is not None and buf[0]:
            _, frame = buf
            aug_frame, output_frame = process_frame(frame, best_model, dataset, display_classes, device, threshold)
            aug_frame = cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR)

            x, y, w, h = 0, 0, legend.shape[1], legend.shape[0]
            output_frame[y:y+h, x:x+w] = cv2.addWeighted(output_frame[y:y+h, x:x+w], 0.2, legend, 0.8, 0.0)
            stacked_frame = np.hstack((aug_frame, output_frame))

            cv2.imshow(f'{video_path}: press q to quit, space to pause', stacked_frame)
            if output:
                if output_stream is None:
                    output_fps = round(stream.get(cv2.CAP_PROP_FPS), 0)
                    output_stream = cv2.VideoWriter(output, FOURCC, output_fps,
                                                    tuple(reversed(stacked_frame.shape[:2])))
                output_stream.write(stacked_frame)
            
            match cv2.waitKey(1) & 0xFF:
                case 113:   # q
                    break

                case 32:    # space
                    while cv2.waitKey(10) & 0xFF != 32:
                        continue

        if output_stream is not None:
            output_stream.release()
            print(f'Video saved to {output}')

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
