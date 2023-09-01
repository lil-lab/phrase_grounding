# Standard Library Imports
import difflib
import argparse
from collections import defaultdict
import textwrap

# Third-Party Imports
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local Module Imports
from src.preprocessing.touchdown_sdr.graph_loader import GraphLoader
from src.preprocessing.touchdown_sdr.equirec2perspec import Equirectangular
from src.preprocessing.touchdown_sdr.loader import Loader

# Image Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Equirectangular Configuration
equ = Equirectangular()

# Image Constants
ORIGINAL_PANO_HEIGHT = 1500
ORIGINAL_PANO_WIDTH = 3000
NUM_PERSPECTIVE_IMAGES = 8
PERSPECTIVE_IMAGE_HEIGHT =  800                                                    
PERSPECTIVE_IMAGE_WIDTH =  3712


def get_optimal_font_scale(text, width=3000):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10

def get_overlap_nonoverlap(s1, s2):
    # https://stackoverflow.com/questions/51059536/python-difflibs-sequencematcher-does-not-find-longest-common-substrings
    s = difflib.SequenceMatcher(None, s1, s2, autojunk=False)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    nonoverlap = s1[pos_a+size:]
    overlap = s1[pos_a:pos_a+size]
    if overlap == s2[-size:]:
        return overlap, nonoverlap
    else:
        return "", s1


def write_discription(img, overlap_text, nonoverlap_text, overlap_color=(0, 0, 255), nonoverlapcolor=(0, 255, 0), textwidth = 55):
    full_text =  overlap_text + nonoverlap_text
    wrapped_texts = textwrap.wrap(full_text, width=textwidth, replace_whitespace=False, drop_whitespace=False)

    # params
    text_counter = 0
    is_overlap = True if len(overlap_text) != 0 else False
    x=100
    y=100
    font_size=3
    font_thickness=3
    font=cv2.LINE_AA

    # estimate the gap
    textsize = cv2.getTextSize(wrapped_texts[0], font, font_size, font_thickness)[0]
    gap = textsize[1] + 10
    total_slack = len(wrapped_texts) * gap
    total_slack += x
    text_img = np.zeros((total_slack, 3000, 3))
    img = np.concatenate([text_img, img], 0)
    img = cv2.merge([img[...,2], img[...,1], img[...,0]])

    # write texts
    for i, line in enumerate(wrapped_texts):
        if is_overlap and (text_counter + textwidth) > len(overlap_text):
            end = len(overlap_text) - text_counter
            overlap_line, nonoverlap_line = line[:end], line[end:] 
            is_overlap = False
            textsize = cv2.getTextSize(overlap_line, font, font_size, font_thickness)[0]
            gap = textsize[1] + 10
            cv2.putText(img, overlap_line, (x, y + i * gap), font,
                        font_size, 
                        overlap_color, 
                        font_thickness, 
                        lineType = cv2.LINE_AA)
            cv2.putText(img, nonoverlap_line, (x + textsize[0] + 1, y + i * gap), font,
                        font_size, 
                        nonoverlapcolor, 
                        font_thickness, 
                        lineType = cv2.LINE_AA)

        else:
            color = overlap_color if is_overlap else nonoverlapcolor
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            gap = textsize[1] + 10
            cv2.putText(img, line, (x, y + i * gap), font,
                        font_size, 
                        color, 
                        font_thickness, 
                        lineType = cv2.LINE_AA)
        text_counter += len(line)
    
    return img, total_slack


class CacheLoader():
    def __init__(self, image_dir, graph_dir):
        super().__init__()
        self._image_dir = image_dir
        self._graph_dir = graph_dir
        self.graph = GraphLoader().construct_graph()
        self.load_yaw_offsets()

    def load_states(self):
        nodes_path = '{}/nodes.txt'.format(self._graph_dir)
        links_path = '{}/links.txt'.format(self._graph_dir)
        self.nodeid_to_panoid = {}
        states = []
        with open(nodes_path) as f:
            for line in f:
                nodeid, panoid, pano_orientation, _, _ = line.split(',')
                self.nodeid_to_panoid[nodeid] = panoid

        with open(links_path) as f:
            for line in f:
                node1, heading, node2, _ = line.split(',')
                image_path = 'panos/{}.jpg'.format(self.nodeid_to_panoid[node1])
                states.append((image_path, int(heading), self.nodeid_to_panoid[node1]))
        return states

    def load_state_dict(self):
        states = self.load_states()
        state_dict = {}
        for image_path, heading, panoid in states:
            state_dict[panoid] = (heading, image_path)
        return state_dict
    
    def load_yaw_offsets(self):
        nodes_path = "{}/nodes.txt".format(self.graph_dir)
        self.yaw_angles = {}
        with open(nodes_path) as f:
            for line in f:
                panoid, pano_yaw_angle, lat, lng = line.strip().split(',')
                self.yaw_angles[panoid] = float(pano_yaw_angle)

    def load_pano_from_s3(self, data):
        for i, data_obj in enumerate(data):
            s3_obj = bucket.Object(data_obj['image_file'])
            stream = io.BytesIO()
            s3_obj.download_fileobj(stream)
            image_obj = c(stream)
            image_obj = image_obj.resize((3000, 1500))
            image = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2BGR)
            yield image, float(data_obj['heading']), data_obj['td_location_text']

        shift_angle = self.yaw_angles[data_obj["panoid"]] -  heading
  
  
    def get_shift_angles(self, data_obj):
        if data_obj["prev_panoid"] is not None:
            heading = self.graph.edges[(data_obj["prev_panoid"], data_obj["panoid"])]
            in_restricted_set = True
        elif data_obj["panoid"] == data_obj["route_panoids"][-2]:
            heading = self.graph.edges[(data_obj["panoid"], data_obj["route_panoids"][-1])]
            in_restricted_set = True
            if data_obj["heading"] == "":
               data_obj["heading"] = 0 
        else:
            try:
                heading = float(data_obj["heading"])
            except:
                print("{}, {} missing heading".format(data_obj["route_id"], data_obj["panoid"]))
                heading = 0
                data_obj["heading"] = 0
            in_restricted_set = False

        if data_obj["panoid"] in self.yaw_angles:
            shift_angle = self.yaw_angles[data_obj["panoid"]] -  heading
            td_shift_angle = self.yaw_angles[data_obj["panoid"]] -  float(data_obj["heading"])
        else:
            print("{} missing yaw_angles".format(data_obj["panoid"]))
            shift_angle, td_shift_angle = 0, 0 

        # perspective transformation angles
        angles = np.array([225, 270, 315, 0, 45, 90, 135, 180])
        angles = angles - 22.5
        angles = angles - shift_angle
        return angles, shift_angle, td_shift_angle, in_restricted_set


    def save_perspective_images(self, data, out_image_dir, debug: bool = False):

        for i, data_obj in enumerate(tqdm(data, disable=False)):
            image_path = os.path.join(self._image_dir, "{}.jpg".format(data_obj["panoid"]))
            if not os.path.exists(image_path):
                print("{} missing".format(image_path))
                continue

            # calculate angles
            angles, shift_angle, td_shift_angle, in_restricted_set = self.get_shift_angles(data_obj)

            image_obj = Image.open(image_path)
            pano = np.array(image_obj)

            if debug:
                counter =  0

                # save rolled images
                np_image = np.array(image_obj)
                width = np_image.shape[1]
                shift = int(width * shift_angle / 360)
                rolled_np_image = np.roll(np_image, shift, axis=1)
                rolled_image = Image.fromarray(np.uint8(rolled_np_image)).convert('RGB')
                rolled_img_save_path = "{}/{}_{}_rolled_pano.jpg".format(out_image_dir, data_obj["route_id"], data_obj["panoid"])

                # save touchdown images
                td_shift = int(width * td_shift_angle / 360)
                image_obj = Image.open(image_path)
                np_image = np.array(image_obj)
                width = np_image.shape[1]
                rolled_td_np_image = np.roll(np_image, td_shift, axis=1)
                rolled_img_td_save_path = "{}/{}_{}_rolled_td.jpg".format(out_image_dir, data_obj["route_id"], data_obj["panoid"])
                rolled_image =  cv2.merge([rolled_td_np_image[...,2], rolled_td_np_image[...,1], rolled_td_np_image[...,0]])
                rolled_image = cv2.circle(rolled_image, tuple([int(3000/2) , int (1500/2)]), 40, (255, 0, 0), thickness=4)

                # stack images
                stacked_img_save_path = "{}/{}_{}_stacked.jpg".format(out_image_dir, data_obj["route_id"], data_obj["panoid"])
                stacked_images = np.concatenate([np_image, rolled_np_image, rolled_td_np_image], 0)
                print(stacked_img_save_path)

                # stack images and write text
                overlap_text, nonoverlap_text = get_overlap_nonoverlap(data_obj["text"], data_obj["nav_text"])
                stacked_images, top_offset = write_discription(stacked_images, overlap_text, nonoverlap_text)

                # center line
                color = (0, 255, 0)
                stacked_images = cv2.line(stacked_images, (1500, top_offset), (1500, stacked_images.shape[0]), color, thickness=3)

                # shift angle
                stacked_images = cv2.putText(stacked_images, "Navigation (shift: {} degrees)".format(np.round(shift_angle, 3)), (100, top_offset + 100 +1500), cv2.FONT_HERSHEY_SIMPLEX, 
                3, (0, 0, 255), 4, cv2.LINE_AA)
                stacked_images = cv2.putText(stacked_images, "MTurker (shift: {} degrees)".format(np.round(td_shift_angle, 3)), (100, top_offset + 100 +1500*2), cv2.FONT_HERSHEY_SIMPLEX, 
                3, (0, 0, 255), 4, cv2.LINE_AA)

                # annotate touchdown location
                target_pano = np.zeros((ORIGINAL_PANO_HEIGHT, ORIGINAL_PANO_WIDTH, 3))
                x = int(data_obj["center"]['x'] * ORIGINAL_PANO_WIDTH)
                y = int(data_obj["center"]['y'] * ORIGINAL_PANO_HEIGHT)
                stacked_images = cv2.circle(stacked_images, tuple([x, y + top_offset]), 40, (0, 0, 255), thickness=4)
                target_pano[y, x, :] = 255

                rolled_target_pano = np.roll(target_pano, shift, axis=1)
                x,y,z= np.unravel_index(rolled_target_pano.argmax(), rolled_target_pano.shape)
                stacked_images = cv2.circle(stacked_images, tuple([y, x + top_offset + 1500]), 40, (0, 0, 255), thickness=4)

                rolled_td_target_pano = np.roll(target_pano, td_shift, axis=1)
                x,y,z= np.unravel_index(rolled_td_target_pano.argmax(), rolled_target_pano.shape)
                stacked_images = cv2.circle(stacked_images, tuple([y, x + top_offset + 1500*2]), 40, (0, 0, 255), thickness=4)

                # save images
                cv2.imwrite(stacked_img_save_path, stacked_images)

            perspective_images = []
            imagefeat_list = []

            for angle_idx, angle in enumerate(angles):
                perspective_image = equ.GetPerspective(pano, FOV, angle, 0, PERSPECTIVE_IMAGE_HEIGHT, PERSPECTIVE_IMAGE_WIDTH)
                pil_perspective_image = Image.fromarray(perspective_image)
                image = transform(pil_perspective_image).to(device)
                perspective_images.append(perspective_image)

                if calc_feat:
                    image_feat = resnet(image.unsqueeze(0)).squeeze(0)
                    image_feat = image_feat.to(torch.device('cpu')).numpy()
                    imagefeat_list.append(image_feat)

            perspective_images = np.concatenate(perspective_images, 1)
            pil_perspective_images = Image.fromarray(np.uint8(perspective_images)).convert('RGB')
            img_save_path = "{}/{}_{}.jpg".format(out_image_dir, data_obj["route_id"], data_obj["panoid"])
            pil_perspective_images.save(img_save_path)

            if debug:
                counter += 1
                if counter > 10:
                    print(counter)
                    break


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--out_alias', type=str, default='', help='experimental name to be output in wandb')
    parser.add_argument('--json_dir', type=str, default='', help='the directory containing images')
    parser.add_argument('--image_dir', type=str, default='', help='the directory containing images')
    parser.add_argument('--graph_dir', type=str, default='', help='the directory containing graphs of streetview')

    args = parser.parse_args()

    loader = Loader(args.json_dir, None, None)
    cache_loader = CacheLoader(args.image_dir, args.graph_dir)

    # Debug
    data = loader.load_json('debug_dev.json')
    cache_loader.save_perspective_images(data, out_image_dir)

    # Train data
    data = loader.load_json('train.json')
    cache_loader.save_perspective_images(data, out_image_dir)

    # Dev data
    data = loader.load_json('dev.json')
    cache_loader.save_perspective_images(data, out_image_dir)

    # Test data
    data = loader.load_json('test.json')
    cache_loader.save_perspective_images(data, out_image_dir)

