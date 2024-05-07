from flask import Flask, request, Response, send_file
from XAI.xai import generate_xai
import os

app = Flask(__name__)
IMAGES_DIRECTORY = 'XAI_Results/XAI'

@app.route('/process_mri', methods=['POST'])
def do_xai():
    mri_volume = request.data  # Access the MRI volume data
    # Call the function to process the MRI volume
    generate_xai(mri_volume)
    
@app.get('/get_images')
async def get_images():
    image_files = [os.path.join(IMAGES_DIRECTORY, filename) for filename in os.listdir(IMAGES_DIRECTORY)]
    image_data = []
    for image_file in image_files:
        with open(image_file, 'rb') as f:
            image_data.append(f.read())
    return Response(content=image_data, media_type="image/png")
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app
