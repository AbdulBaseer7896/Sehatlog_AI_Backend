
from utilits.Groq import GroqLLM
from mistralai import Mistral
import base64
import os



def extrict_Data_From_image_using_OCR(file):
        mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        base64_image = base64.b64encode(file.read()).decode('utf-8')
        
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/{file.content_type.split('/')[-1]};base64,{base64_image}"
            }
        )

        print("This is the ocr response = " , ocr_response)

        return ocr_response
