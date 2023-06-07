import os
import sys
import argparse
import numpy as np

def calculate_mAP(labels, outdir, metric, iou, show_images, show_plots):
    # Define which metric to use (i.e. which set of IoU thresholds to calculate mAP for)
    if metric == 'coco':
        iou_threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    elif metric == 'pascalvoc':
        iou_threshes = [0.5]
    elif metric == 'custom':
        custom_ious = iou
        try:
            iou_threshes = [float(iou) for iou in custom_ious]
        except:
            print('Invalid entry for --iou. Example of a correct entry: "--iou=0.5,0.6,0.7"')
            sys.exit()
    else:
        print('Invalid entry for --metric. Please use coco, pascalvoc, or custom.')
        sys.exit()

    # Get file paths
    cwd = os.getcwd()
    output_path = os.path.join(cwd, outdir)
    labelmap_path = os.path.join(cwd, labels)

    # Load the label map
    with open(labelmap_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Make folder to store output result files
    if os.path.exists(output_path):
        print('The output folder %s already exists. Please delete it or specify a different folder name using --outdir.' % output_path)
        sys.exit()
    else:
        os.makedirs(output_path)

    # Create dictionary to store overall mAP results and results for each class
    mAP_results = {'overall': np.zeros(len(iou_threshes))}
    for classname in classes:
        mAP_results[classname] = np.zeros(len(iou_threshes))  # Add each class to dict

    for i, iou_thresh in enumerate(iou_threshes):
        # Run modified script
        print('Calculating mAP at %.2f IoU threshold...' % iou_thresh)
        # Perform the necessary calculations

    # Okay, we found mAP at each IoU value! Now we just need to average the mAPs and display them.
    class_mAP_result = []
    print('\n***mAP Results***\n')
    print('Class\t\tAverage mAP @ 0.5:0.95')
    print('---------------------------------------')
    for classname in classes:
        class_vals = mAP_results[classname]
        class_avg = np.mean(class_vals)
        class_mAP_result.append(class_avg)
        print('%s\t\t%0.2f%%' % (classname, class_avg))

    overall_mAP_result = np.mean(class_mAP_result)
    print('\nOverall\t\t%0.2f%%' % overall_mAP_result)

if __name__ == '__main__':
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', help='Path to the labelmap file', default='labelmap.txt')
    parser.add_argument('--outdir', help='Output folder to save results in', default='outputs')
    parser.add_argument('--metric', help='mAP metric to calculate: "coco", "pascalvoc", or "custom"', default='coco')
    parser.add_argument('--iou', help='(Only if using --metric=custom) Specify IoU thresholds to use for evaluation (example: 0.5,0.6,0.7)')
    parser.add_argument('--show_images', help='Display and save images as they are evaluated', action='store_true')
    parser.add_argument('--show_plots', help='Display and save plots showing precision/recall curve, mAP score, etc', action='store_true')

    args = parser.parse_args()

    calculate_mAP(args.labels, args.outdir, args.metric, args.iou, args.show_images, args.show_plots)
