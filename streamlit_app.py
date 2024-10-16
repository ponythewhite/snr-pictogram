from io import BytesIO

import cairosvg
import numpy as np
import streamlit as st
from PIL import Image


# Function to load the reference image
def load_image(image_file):
  """Load an image, handle alpha channel, and convert it to grayscale."""
  image = Image.open(image_file)
  if image.mode in ('RGBA', 'LA'):
    # Handle transparency by compositing over a white background
    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image.convert('RGBA'))
  image = image.convert('L')
  return image


# Function to preprocess the image
def preprocess_image(image, block_size):
  """Crop the image so that its dimensions are multiples of the block size."""
  width, height = image.size
  new_width = (width // block_size) * block_size
  new_height = (height // block_size) * block_size
  image = image.crop((0, 0, new_width, new_height))
  return image


# Function to load pictograms from uploaded files
def load_pictograms(pictogram_files, block_size):
  """Load SVG pictograms from file-like objects, render them to images at the specified size."""
  pictograms = []
  for file in pictogram_files:
    # Read the SVG data
    svg_data = file.read()
    # Convert SVG to PNG image data at the specified size
    png_data = cairosvg.svg2png(bytestring=svg_data,
                                output_width=block_size,
                                output_height=block_size)
    # Load image data into PIL Image
    img = Image.open(BytesIO(png_data)).convert('RGBA')
    pictograms.append(img)
    # Reset file pointer to start
    file.seek(0)
  return pictograms


# Function to create grayscale pictograms
def create_grayscale_pictograms(pictograms):
  """Create grayscale versions of pictograms for matching."""
  pictograms_gray = []
  for img in pictograms:
    # Create grayscale version
    img_gray = Image.alpha_composite(
        Image.new('RGBA', img.size, (255, 255, 255, 255)), img).convert('L')
    pictograms_gray.append(np.array(img_gray))
  return pictograms_gray


# Function to create colored pictograms
def create_colored_pictograms(pictograms, green_color=(0, 254, 189)):
  """Create green versions of pictograms."""
  pictograms_green = []
  for img in pictograms:
    datas = img.getdata()
    newData = []
    for item in datas:
      # item is (R, G, B, A)
      if item[3] == 0:
        # Transparent pixel remains transparent
        newData.append((0, 0, 0, 0))
      else:
        # Replace color with specified green color
        newData.append(
            (green_color[0], green_color[1], green_color[2], item[3]))
    img_colored = Image.new('RGBA', img.size)
    img_colored.putdata(newData)
    pictograms_green.append(img_colored)
  return pictograms_green


# Function to match blocks
def match_blocks(input_array, pictograms_gray, input_block_size):
  """Match each block in the input image with the best matching pictogram based on mean intensity."""
  output_indices = np.zeros((input_array.shape[0] // input_block_size,
                             input_array.shape[1] // input_block_size),
                            dtype=np.int32)
  height, width = input_array.shape

  pictogram_features = np.array([p.mean() for p in pictograms_gray])

  for i in range(0, height, input_block_size):
    for j in range(0, width, input_block_size):
      block = input_array[i:i + input_block_size, j:j + input_block_size]
      block_feature = block.mean()
      errors = np.abs(pictogram_features - block_feature)
      min_index = np.argmin(errors)
      output_indices[i // input_block_size, j // input_block_size] = min_index
  return output_indices


# Function to create the output image
def create_output_image(output_indices, pictograms_color, pictograms_green,
                        output_block_size, green_positions_set,
                        num_provided_pictograms):
  """Create the final output image."""
  grid_height, grid_width = output_indices.shape
  output_image = Image.new(
      'RGBA',
      (grid_width * output_block_size, grid_height * output_block_size),
      (0, 0, 0, 0))

  for i in range(grid_height):
    for j in range(grid_width):
      idx = output_indices[i, j]
      if (i, j) in green_positions_set and idx < num_provided_pictograms:
        # Use green version
        pictogram_img = pictograms_green[idx]
      else:
        # Use original version
        pictogram_img = pictograms_color[idx]
      # Paste the pictogram onto the output image
      output_image.paste(pictogram_img,
                         (j * output_block_size, i * output_block_size),
                         pictogram_img)
  return output_image


# Streamlit app starts here
st.title("PictoSNR: Jack's Pictogram Collage Generator")

st.markdown("""
Upload a reference image and a set of pictograms (SVG files). Adjust the parameters as desired, and generate a collage that resembles the original image using your pictograms. No dick pics allowed!
""")

# Upload reference image
reference_image_file = st.file_uploader("Upload Reference Image (JPG or PNG)",
                                        type=['jpg', 'jpeg', 'png'])

# Upload pictogram files
pictogram_files = st.file_uploader("Upload multiple Pictograms (SVG files)",
                                   type=['svg'],
                                   accept_multiple_files=True)

# Parameters
st.header("Parameters")

green_percentage = st.number_input(
    "Green Percentage",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="Percentage of pictograms to color green")

block_sizes_options = [8, 16, 24, 32, 40, 48, 64, 72]
block_sizes = st.multiselect(
    "Block Sizes (small block = many small tiles, large block = few large tiles)",
    block_sizes_options,
    default=block_sizes_options,
    help="Input block sizes for matching")

size_multiplier = st.number_input(
    "Size Multiplier (change output size, upscaling natively from SVD)",
    min_value=0.1,
    value=4.0,
    step=0.1,
    help="Size multiplier for output block size")

# Generate button
if st.button("Generate Collage"):
  if reference_image_file is not None and pictogram_files:
    # Initialize progress bar
    progress_bar = st.progress(0)
    num_steps = len(block_sizes)

    for idx, block_size in enumerate(block_sizes):
      output_block_size = int(size_multiplier * block_size)

      # Load and preprocess the input image
      input_image = load_image(reference_image_file)
      input_image = preprocess_image(input_image, block_size)
      input_array = np.array(input_image)

      # Load input pictograms for matching (size = block_size)
      input_pictograms = load_pictograms(pictogram_files, block_size)
      # Create grayscale versions for matching
      pictograms_gray = create_grayscale_pictograms(input_pictograms)
      num_provided_pictograms = len(pictograms_gray)

      # Add 100% white and 100% black blocks to matching pictograms
      gray_levels = [0, 255]
      for level in gray_levels:
        gray_block = np.full((block_size, block_size), level, dtype=np.uint8)
        pictograms_gray.append(gray_block)

      # Match blocks
      output_indices = match_blocks(input_array, pictograms_gray, block_size)

      # Load output pictograms for output (size = output_block_size)
      output_pictograms = load_pictograms(pictogram_files, output_block_size)
      # Create original color versions
      pictograms_color = output_pictograms.copy()
      # Create green versions
      pictograms_green = create_colored_pictograms(output_pictograms)

      # Add 100% white and 100% black blocks to output pictograms
      for level in gray_levels:
        if level == 255:
          # White block is fully transparent
          img_white = Image.new('RGBA', (output_block_size, output_block_size),
                                (0, 0, 0, 0))
          pictograms_color.append(img_white)
          pictograms_green.append(img_white)
        else:
          # Black block remains black
          img_black = Image.new('RGBA', (output_block_size, output_block_size),
                                (0, 0, 0, 255))
          pictograms_color.append(img_black)
          pictograms_green.append(img_black)

      # Determine which blocks will use the green pictograms
      grid_height, grid_width = output_indices.shape
      provided_pictogram_mask = output_indices < num_provided_pictograms
      provided_positions = np.argwhere(provided_pictogram_mask)
      num_provided_positions = len(provided_positions)
      num_green = int(num_provided_positions * (green_percentage / 100.0))

      # Randomly select positions to be green
      if num_green > 0:
        # np.random.seed(42)  # Uncomment for reproducibility
        green_indices = np.random.choice(num_provided_positions,
                                         num_green,
                                         replace=False)
        green_positions = provided_positions[green_indices]
        green_positions_set = set(map(tuple, green_positions))
      else:
        green_positions_set = set()

      # Create the output image
      output_image = create_output_image(output_indices, pictograms_color,
                                         pictograms_green, output_block_size,
                                         green_positions_set,
                                         num_provided_pictograms)

      # Display the output image
      st.image(output_image,
               caption=f"Output Image (Block Size: {block_size})",
               use_column_width=True)

      # Provide download link
      # Save output image to BytesIO
      img_bytes = BytesIO()
      output_image.save(img_bytes, format='PNG')
      img_bytes.seek(0)

      st.download_button(label=f"Download Image (Block Size: {block_size})",
                         data=img_bytes,
                         file_name=f"output_{block_size}.png",
                         mime="image/png")

      # Update progress bar
      progress_bar.progress((idx + 1) / num_steps)

    st.success("Collage generation complete!")
  else:
    st.warning(
        "Please upload both a reference image and at least one pictogram SVG file."
    )
