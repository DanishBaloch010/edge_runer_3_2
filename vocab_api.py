import os
import ast
import json
import re
import time
import base64
import pdb
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from together import Together
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
load_dotenv()
CORS(app)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


api_key = os.getenv('TOGETHER_API_KEY')

class VocabAPI:
    def __init__(self):
        self.sys_prompt = """
You are an expert in optical character recognition (OCR) and document analysis. Your job is to extract all the relevant text from any image provided, with special attention to format, headings, sub-headings, and body text. Be precise, follow the instructions strictly and return only the text that is present in the image and is asked from you.
"""
        self.user_prompt = """
This is complete vertically stacked image of multiple consecutive ordered pages of an IELTS reading passage. Your task is to carefully analyze the image and extract all the text from it. The passages are long so thats why whole passage is given to you in a form of vertically stiched one picture.


STRICT IMPORTANT NOTE: I NEED EXTRACTED DATA IN THE FORMAT THAT I AM SAYING TO YOU WITHOUT ANY EXPLANATIONS OR SUGGESTIONS FROM YOUR SIDE. I AM A PROGRAMMER AND I WILL USE YOUR OCR EXTRACTED TEXT IN MY APPLICATION. I NEED ONLY REQUIRED EXTRACTED DATA FROM YOU AND NOTHING ELSE.

Some visual cues to assist you in this process:
1- There are most probably instructions too for the students to read the passage(they are mostly in the start of the picture), i also need those instructions.
2- The largest and boldest text at the top of the picture is likely the title of the passage.
3- If there is a subtitle present that will always be below the main title of the passage, although sometimes a subtitle is absent for the passage and not provided by the IELTS.
4- Heading and subtitle are always relevant to each other and they describe an overall idea about the text in the main body of the passage.
5- Following the heading (and potential subtitle), you'll find the main body of the passage. When you are extracting the text from the main body make sure you keep the format of paragraphs the same as depicted in the picture. for every start of new paragraph insert a '\n' (line break), so that my parser later understand that this is the start of a new paragraph. There can be a case where a half portion of the last paragraph is in the first page and the other remaining half is at the start of the second page, which means a parapraph might continue to the next page but it is only one paragraph as a whole.
6- According to official IELTS website the reading passages have 2000-3000 words in the main body as a whole.

Please extract all my mentioned and required details from the image, as I will conduct further analysis on it afterward.

Please deeply analyze the full text of the passage and return it in the following JSON format: 
{
    "instructions": "Instruction by IELTS (if present otherwise write None)",
    "title": "Extracted Title Here",
    "subtitle": "Extracted Subtitle Here (if present otherwise write None)",
    "text": "Full text of the passage here."

}


"""
        self.sys_prompt_vocab = """
You are an expert english teacher. you know about the grammar rules and have extensive depth in english literature. you have to analyze the given words of english and teach them to students in a way that the meaning remain in their head. you are a great, professional, interactive and expert english teacher.
"""
        self.user_prompt_vocab = """
You are an expert English teacher. You know about the grammar rules and have extensive knowledge of English literature. You have to analyze the given words in depth and find relations. Your goal is to provide a list of words that I can use in my IELTS English speaking and writing tasks to improve my vocabulary. Ignore the words in the array that cannot be used in conversations.

NOTE: YOU HAVE TO FOLLOW THE STRICT GUIDELINES GIVEN TO YOU AND RETURN THE REQUIRED ANSWER IN THE SPECIFIED FORMAT AT THE END OF THIS PROMPT. PROVIDE THE DATA IN JSON FORMAT, AND NOTHING ELSE. I WILL USE THE OUTPUT DIRECTLY IN MY APPLICATION, SO IT MUST BE STRICTLY IN JSON.

Specifically, focus on these things:
1 - Group the words that have similar or closely related meanings, and ensure these words can be used in both English speaking and writing. Make at least 10 groups of words, each containing a maximum of 5 words or a minimum of 1.
2 - Generate a vivid meaning of each word that is relevant to the English dictionary.
3 - For speaking, group semi-formal words together.
4 - For writing, group formal words that can be used in formal English writing.
5 - Once the grouping of words is complete, craft two sentences for each word: one for formal academic writing and one for semi-formal speaking.
6 - Also, ensure that the two generated sentences for each word are linked together, so it is easy for students to grasp the meaning of the word and use it in different contexts.

In this way, students can learn more English words intuitively, and it will remain in their heads for a long time.

OUTPUT FORMAT:
Please return the data in JSON format with the following structure:

{
  "groups": [
    {
      "group_number": "<group_id>",
      "words": [
        {
          "word": "<word>",
          "meaning": "<meaning in English literature>",
          "formal_writing_sentence": "<formal academic sentence>",
          "semi_formal_speaking_sentence": "<semi-formal conversational sentence>"
        },
        ...
      ]
    },
    ...
  ]
}

ARRAY OF ENGLISH WORDS:
"""
        
        self.client = Together(base_url="https://api.aimlapi.com/v1", api_key=api_key)
        self.model = 'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo'
    
    def preprocess(self, input_string):
        output = input_string.choices[0].message.content
        output = output.lower()

        file_content = re.sub(r'[^\w\s]', '', output)
        file_content = re.sub(r'\b\d+\b', '', file_content)

        words = word_tokenize(file_content)

        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]

        tagged_words = nltk.pos_tag(filtered_words)
        vocabulary_words = [word for word, pos in tagged_words if pos not in ['NNP', 'CD']]

        vocabulary_words = list(set(vocabulary_words))
        vocabulary_words_string = "['" + "', '".join(vocabulary_words) + "']"
        return vocabulary_words_string

    
    def output_model(self, input_prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
            {
                "role": "system",
                "content": self.sys_prompt,
            },
        
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":  input_prompt,
                    },
                                
                ],
            }
        ],
                temperature=1.0,
                max_tokens=3500
            )
        except Exception as e:
            print(f"Error: {e}")
            return None
        return response
        

    def llama_model(self, image_path, user_prompt):
        """
        Feed local images into the Together API for processing by a multi-modal LLM.
        """
        # with open(image_path, "rb") as image_file:
        #     encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        encoded_images = []
        with open(image_path, "rb") as image_file:
            encoded_image_paragraph = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append(encoded_image_paragraph)


        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
            {
                "role": "system",
                "content": self.sys_prompt,
            },
        
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":  user_prompt,
                    },
                    
                    {
                        "type": "image_url",
                        "image_url": {

                            "url" :f"data:image/png;base64,{encoded_image_paragraph}"

                        },
                    }

                                
                ],
            }
        ],
                temperature=1.0,
                max_tokens=3500
            )
        except Exception as e:
            print(f"Error: {e}")
            return None
        return response


    def start_process(self, image_path):
        # Test feeding a specific local image to the Llama model
        image_path = image_path

        output = self.preprocess(self.llama_model(image_path, self.user_prompt))
        user_prompt = self.user_prompt_vocab + ' ' + output
        response = self.output_model(user_prompt)
        response_text = response.choices[0].message.content
        response = response_text.replace('\n', '')
        if response:
            output = json.loads(response)

        return output
        
    
@app.route('/vocab', methods=['POST'])
def recommend():
    try:
        image = request.files.get('image')

        # Ensure image is provided
        if not image:
            return jsonify({"error": "Please upload an image"}), 400

        # Save image temporarily and get the path
        image_path = f"/tmp/{image.filename}"
        image.save(image_path)

        # Pass the image path to start_process
        obj = VocabAPI()
        extracted_text = obj.start_process(image_path)  # Pass single image path in a list
        print(extracted_text)

        return jsonify({"vocab": extracted_text}), 200

    except Exception as e:
        logging.exception(f"Error in recommendation process: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
