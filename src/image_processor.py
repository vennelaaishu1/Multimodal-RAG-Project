import io
import base64
from google import genai


def process_image(image, question, api_key):

    client = genai.Client(
        api_key=api_key
    )

    # Convert PIL image to bytes
    image_bytes_io = io.BytesIO()

    image.save(
        image_bytes_io,
        format="JPEG"
    )

    image_bytes = image_bytes_io.getvalue()

    # Convert image bytes to Base64
    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=[
            {
                "type": "text",
                "text": question
            },
            {
                "type": "image",
                "data": image_base64,
                "mime_type": "image/jpeg"
            }
        ]
    )

    return interaction.output_text