import os
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from datetime import datetime
import gzip
from tqdm import tqdm

class HierTextGenerator:
    def __init__(self, character_map, font_paths, font_sizes, backgrounds=None,
                 apply_rotation=False, apply_blur=False, width_padding_factor=1.0, height_padding_factor=1.0, 
                 debug=False):
        self.character_map = character_map
        self.font_paths = font_paths
        self.font_sizes = font_sizes
        self.backgrounds = backgrounds
        self.apply_rotation = apply_rotation
        self.apply_blur = apply_blur
        self.width_padding_factor = width_padding_factor
        self.height_padding_factor = height_padding_factor
        self.debug = debug

    def rotate_point(self, x, y, cx, cy, angle_degrees):
        """Rotates a point (x, y) around a center (cx, cy) by an angle in degrees."""
        angle_radians = math.radians(-1 * angle_degrees)
        cos_theta = math.cos(angle_radians)
        sin_theta = math.sin(angle_radians)

        # Translate point to origin relative to center
        temp_x = x - cx
        temp_y = y - cy

        # Rotate point
        rotated_x = temp_x * cos_theta - temp_y * sin_theta
        rotated_y = temp_x * sin_theta + temp_y * cos_theta

        # Translate point back
        new_x = int(rotated_x + cx)
        new_y = int(rotated_y + cy)

        return new_x, new_y

    def rotate_bbox_corners(self, corners, center, angle_degrees):
        cx, cy = center
        return [self.rotate_point(x, y, cx, cy, angle_degrees) for (x, y) in corners]

    def generate_paragraph(self, max_words=5):
        """
        Generate a paragraph-like structure by mixing words from the character-to-words map.
        """
        paragraph = []
        for _ in range(max_words):
            paragraph.append(random.choice(self.character_map[random.choice(list(self.character_map.keys()))]))
        return " ".join(paragraph)

    # Function to generate an image with text
    def generate_image_with_text(self, text, font, font_size, image_size=(2, 2), text_color=(0, 0, 0), padding=10):
        """
        Generate an image with the given text. Dynamically adjusts the image size to fit the text if necessary.
        """
        # Create a temporary image to calculate text size
        temp_image = Image.new("RGBA", image_size, color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_image)
        text_width = draw.textlength(text, font=font)
        text_height = font_size
        
        # Adjust image size if text is larger than the default size
        adjusted_width = int(max(image_size[0], text_width + 2 * padding))
        adjusted_height = int(max(image_size[1], text_height + 2 * padding))
        image = Image.new("RGBA", (adjusted_width, adjusted_height), color=(0, 0, 0, 0))
        
        # Recalculate text position
        draw = ImageDraw.Draw(image)
        position = ((adjusted_width - text_width) // 2, (adjusted_height - text_height) // 2)
        
        # Draw the text
        draw.text(position, text, fill=text_color, font=font)
        return image

    # Function to apply transformations to an image
    def apply_transformations(self, image, rotation=None, blur_radius=None, background=None):
        """
        Apply transformations like rotation and blur to the image.
        If no transformations are specified, the image is returned as-is.
        """
        # Rotate the image if rotation is specified
        if rotation is not None:
            image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)
        
        # Blur if needed
        if blur_radius is not None and blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Add random background if backgrounds are provided
        if background:
            # Resize background to match the (possibly expanded) image size
            background = background.resize(image.size)
            # Paste the text image (with transparency) onto the background
            if image.mode == "RGBA":
                background.paste(image, (0, 0), image)
            else:
                background.paste(image, (0, 0))
            image = background

        return image

    def create_hiertext_structure(self, images, output_dir, annotations, prefix):
        """
        Create a HierText-like structure for the dataset and save it as a GZIP-compressed JSONL file.
        """
        dataset = {
            "info": {
                "date": datetime.today().strftime('%Y-%m-%d'),
                "version": "1.0"
            },
            "annotations": []
        }
        
        # Create the 'gt' folder for the JSON file
        gt_folder = os.path.join(output_dir, "gt")
        os.makedirs(gt_folder, exist_ok=True)
        
        # Create the folder for images, named after the prefix
        images_folder = os.path.join(output_dir, prefix)
        os.makedirs(images_folder, exist_ok=True)
        
        for idx, (image, annotation) in enumerate(zip(images, annotations)):
            image_width, image_height = image.size
            
            words_metadata = []
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            # Process each word in the paragraph
            draw = ImageDraw.Draw(image)
            font_size = annotation.pop("font_size")
            font_path = annotation.pop("font_path")
            font = ImageFont.truetype(
                font_path, 
                font_size
            )
            x, y = annotation["position"]

            # Prepare spacing and height
            whitespace_width = draw.textlength(" ", font=font)
            width_padding = int(whitespace_width * self.width_padding_factor)
            for word in annotation["text"].split():
                word_offset = (x, y)
                (left, top, right, bottom) = draw.textbbox(word_offset, word, font=font)
                
                text_height =  bottom - top
                height_padding = int(text_height * self.height_padding_factor)           
                
                word_vertices = [
                    [left - width_padding, top - height_padding],
                    [right + width_padding, top - height_padding],
                    [right + width_padding, bottom + height_padding],
                    [left - width_padding, bottom + height_padding]
                ]

                # Roteate the bounding box if rotation was applied
                if "rotation" in annotation:
                    rotation = annotation["rotation"]
                    center = annotation["center"]
                    word_vertices = self.rotate_bbox_corners(word_vertices, center, rotation)
                    
                if self.debug:
                    # For debugging, draw the word bounding boxes on the image
                    draw.polygon(word_vertices, outline="blue", fill=None)

                # Convert points to integers
                word_vertices = [[int(x) for x in point] for point in word_vertices]

                words_metadata.append({
                    "vertices": word_vertices,
                    "text": word,
                    "legible": True,
                    "handwritten": False,
                    "vertical": False
                })
                
                # Update bounding box for the paragraph
                min_x = min(min_x, left - width_padding)
                min_y = min(min_y, top - height_padding)
                max_x = max(max_x, right + width_padding)
                max_y = max(max_y, bottom + height_padding)
                
                x = right
                x += whitespace_width
            
            # Calculate paragraph bounding rectangle
            paragraph_vertices = [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ]

            # Rotate the paragraph bounding box if rotation was applied
            if "rotation" in annotation:
                rotation = annotation["rotation"]
                center = annotation["center"]
                paragraph_vertices = self.rotate_bbox_corners(paragraph_vertices, center, rotation)

            if self.debug:
                # Debug paragraph bounding box
                draw.polygon(paragraph_vertices, outline="red", fill=None)

            # Convert points to integers
            paragraph_vertices = [[int(x) for x in point] for point in paragraph_vertices]

            # Save the image with a unique filename
            image_filename = f"{prefix}_{idx + 1}.jpg"
            image_path = os.path.join(images_folder, image_filename)
            image.save(image_path)
            
            # Add annotation metadata
            dataset["annotations"].append({
                "image_id": image_filename.rsplit('.', 1)[0],  # Remove the file extension
                "image_width": image_width,
                "image_height": image_height,
                "paragraphs": [
                    {
                        "vertices": paragraph_vertices,  # Bounding rectangle for the entire paragraph
                        "legible": True,
                        "lines": [
                            {
                                "vertices": paragraph_vertices,  # Bounding rectangle for the line
                                "text": annotation["text"],
                                "legible": True,
                                "handwritten": False,
                                "vertical": False,
                                "words": words_metadata  # List of words with their bounding boxes
                            }
                        ]
                    }
                ]
            })
        
        # Save the dataset as a GZIP-compressed JSONL file
        jsonl_path = os.path.join(gt_folder, f"{prefix}.jsonl.gz")
        with gzip.open(jsonl_path, "wt", encoding="utf-8") as gz_file:
            gz_file.write(json.dumps(dataset, indent=4))  # Write the entire dataset as a single JSON object
        
        print(f"Dataset saved to {jsonl_path}")
        print(f"Images saved to {images_folder}")

    def generate_dataset(self, output_dir, prefix, num_samples=10):
        """
        Generate a dataset in the HierText format with paragraph-like structures.
        By default, no transformations are applied.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images = []
        annotations = []
        
        for i in range(num_samples):
            # Set font and size
            font_size = random.choice(self.font_sizes)
            font_path = random.choice(self.font_paths)
            font = ImageFont.truetype(
                font_path, 
                font_size
            )
            
            # Set background and text color
            if self.backgrounds:
                bg = random.choice(self.backgrounds)
                bg_color = bg["color"]
                bg_path = bg["path"]
                if len(bg_path) > 0:
                    background = Image.open(bg_path).convert("RGB")
                elif len(bg_color) > 0:
                    background = Image.new("RGB", (400, 400), bg_color)
                text_color = (255, 255, 255) if bg["is_dark"] else (0, 0, 0)
            else:
                background = None
            
            # Generate a paragraph
            paragraph = self.generate_paragraph()
            image = self.generate_image_with_text(paragraph, font, font_size, text_color=text_color)
            
            # Apply transformations if desired
            rotation = random.randint(-45, 45) if self.apply_rotation else None
            blur_radius = random.uniform(0, 1) if self.apply_blur else None
            image = self.apply_transformations(image, rotation=rotation, blur_radius=blur_radius, background=background)
            
            # Calculate text position
            draw = ImageDraw.Draw(image)
            text_width = draw.textlength(paragraph, font=font)
            text_height = font_size
            position = ((image.width - text_width) // 2, (image.height - text_height) // 2)
            
            images.append(image)
            annotations.append({
                "text": paragraph,
                "font_path": font_path,
                "font_size": font_size,
                "position": position,
                "center": (image.width // 2, image.height // 2),
                "rotation": rotation if rotation is not None else 0
            })
        
        self.create_hiertext_structure(images, output_dir, annotations, prefix)

    @classmethod
    def from_config(cls, config_path, debug=False):
        """
        Create a HierTextGenerator instance from a TOML configuration file.
        """
        import tomllib
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)

        # Create mappings for character to word
        charcter_word_map = {char_map_obj["scoped_character"]: char_map_obj["word_list"] for char_map_obj in cfg["character_maps"]}

        return cls(
            charcter_word_map,
            cfg["fonts"],
            cfg["font_sizes"],
            backgrounds=cfg.get("backgrounds"),
            apply_rotation=cfg.get("apply_rotation", False),
            apply_blur=cfg.get("apply_blur", False),
            width_padding_factor=cfg.get("width_padding_factor", 1.0),
            height_padding_factor=cfg.get("height_padding_factor", 1.0),
            debug=debug
        )