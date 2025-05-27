#!/bin/bash

# Test curl commands for Bagel-DFloat11 Flask server
# These commands assume the server is running on localhost:5000

echo "Testing the server endpoints..."

# 1. Test the /ping endpoint
echo -e "\n=== Testing /ping endpoint ==="
curl -X GET "http://localhost:5000/ping"
echo -e "\n"

# 2. Test the /generate endpoint with basic prompt
echo -e "\n=== Testing /generate endpoint with basic prompt ==="
echo "Generating image with prompt 'A beautiful sunset over mountains'..."
curl -X GET "http://localhost:5000/generate?prompt=A%20beautiful%20sunset%20over%20mountains" --output basic_image.png
echo "Image saved as basic_image.png"

# 3. Test the /generate endpoint with advanced parameters
echo -e "\n=== Testing /generate endpoint with advanced parameters ==="
echo "Generating image with custom parameters..."
curl -X GET "http://localhost:5000/generate?prompt=A%20futuristic%20city%20with%20flying%20cars&show_thinking=true&cfg_text_scale=5.0&cfg_interval=0.5&timestep_shift=2.5&num_timesteps=60&seed=42&image_ratio=16:9" --output advanced_image.png
echo "Image saved as advanced_image.png"

# 4. Test with URL-encoded complex prompt
echo -e "\n=== Testing with complex prompt ==="
echo "Generating image with a complex prompt..."
curl -X GET "http://localhost:5000/generate?prompt=A%20magical%20forest%20with%20glowing%20mushrooms%2C%20fairies%20flying%20around%2C%20and%20a%20small%20cottage%20in%20the%20distance&cfg_text_scale=6.0&image_ratio=4:3" --output complex_image.png
echo "Image saved as complex_image.png"

echo -e "\n=== All tests completed ==="

# DOCUMENTATION:
# 
# ENDPOINT: /ping
# METHOD: GET
# PARAMETERS: None
# DESCRIPTION: Simple health check endpoint that returns {"status": "ok"}
#
# ENDPOINT: /generate
# METHOD: GET
# PARAMETERS:
#   - prompt (required): Text description of the image to generate
#   - show_thinking (optional, default: false): Whether to show the thinking process
#   - cfg_text_scale (optional, default: 4.0): Controls how strongly the model follows the text prompt
#   - cfg_interval (optional, default: 0.4): Start of CFG application interval
#   - timestep_shift (optional, default: 3.0): Higher values for layout, lower for details
#   - num_timesteps (optional, default: 50): Total denoising steps
#   - cfg_renorm_min (optional, default: 1.0): 1.0 disables CFG-Renorm
#   - cfg_renorm_type (optional, default: 'global'): Type of CFG renormalization
#   - max_think_token_n (optional, default: 1024): Maximum number of tokens for thinking
#   - do_sample (optional, default: false): Enable sampling for text generation
#   - text_temperature (optional, default: 0.3): Controls randomness in text generation
#   - seed (optional, default: 0): Seed for reproducibility
#   - image_ratio (optional, default: '1:1'): Aspect ratio of the generated image (options: 1:1, 4:3, 3:4, 16:9, 9:16)
# DESCRIPTION: Generates an image based on the provided text prompt and returns it as a PNG file
#
# USAGE EXAMPLES:
# 1. Basic usage: curl -X GET "http://localhost:5000/generate?prompt=A%20beautiful%20sunset" --output image.png
# 2. With parameters: curl -X GET "http://localhost:5000/generate?prompt=A%20futuristic%20city&cfg_text_scale=5.0&seed=42" --output image.png
#
# NOTE: Make sure to URL-encode the prompt parameter, especially if it contains spaces or special characters.