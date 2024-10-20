import os
import ast
import json
import re
import time
import base64
import pdb
from together import Together
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('TOGETHER_API_KEY')

class HackathonProduct:
    def __init__(self):
        self.sys_prompt = """
You are an expert in optical character recognition (OCR) and document analysis. Your job is to extract all the relevant text from any image provided, with special attention to format, headings, sub-headings, and body text. Be precise, follow the instructions strictly and return only the text that is present in the image and is asked from you. 
"""
        self.user_prompt_paragraph = """
You will be provided with image containing paragraphs. Your task is as follows:

Instructions:
1. Extract and return all paragraphs from the image.
2. Maintain the exact formatting as shown in the image.
3. Output strictly in JSON format.

Output format:
{
  "paragraphs": "<Extracted paragraph 1>",
}

"""
        self.user_prompt_questions = """
You will be provided with image containing questions. Your task is as follows:

1. Identify Questions:
   - Identify the instructions and ignore them.
   - Extract only the questions, ignoring any instructions or rules given.
   - If a set of rules (e.g., "TRUE/FALSE/NOT GIVEN") is provided before the questions, exclude the rules and focus solely on the questions.
   - Identify each question type based on instructions given.
   - Strictly remove the instructions from output and fetch questions only

2. Numbering Questions:
   - Identify the index of each question, it mainly starts from 1.
   - Continue numbering sequentially across multiple images if questions span across multiple images.
   - Return the extracted text in the following JSON format without providing any explanation.

3. Output Format:
  
   {
    "Questions": {
    "1": "<Extracted question 1>",
    "2": "<Extracted question 2>",
    "...": "<Extracted question N>"
    }
   }
"""
        self.user_prompt_solutions = """
You will be provided with image containing solutions. Your task is as follows:

1. Identify Answers:
   - Extract only the answers starting from index 1 if given, ignoring any instructions or rules provided.

2. Numbering Answers:
   - Identify the index to each answers.
   - Continue numbering sequentially.
   - Return the extracted text in the following JSON format without providing any explanation.

3. Output Format:

   {
        "1": "<Extracted answer 1>",
        "2": "<Extracted answer 2>",
        "...": "<Extracted answer N>"
    }
   
"""
        self.actual_prompt = """
You are an expert in text analysis and comprehension. You will be provided with a paragraph, a set of questions, and corresponding answers separated by "<===>". Your task is to analyze the provided text and fulfill the following requirements:

1. Extract the full paragraph and identify its structure, noting the specific locations of information (e.g., "paragraph 3, line 4").

2. For each question, provide:
   - The question index (e.g., 1, 2, 3, etc.).
   - The corresponding answer from the solution.
   - A clear and concise reasoning for why that specific answer was selected, referencing relevant parts of the paragraph.
   - The location of the relevant information in the paragraph (e.g., "paragraph 2, line 3").
   - Return the extracted text in the following JSON format:

3. Output Format:
  
{
  "answers": {
    "1": {
      "answer": "<Corresponding answer for question 1>",
      "reasoning": "<Reason for the answer based on the paragraph>",
      "location": "<Location in the paragraph (e.g., 'paragraph 2, line 3')>"
    },
    "2": {
      "answer": "<Corresponding answer for question 2>",
      "reasoning": "<Reason for the answer based on the paragraph>",
      "location": "<Location in the paragraph (e.g., 'paragraph 3, line 5')>"
    },
    "...": {
      "answer": "<Corresponding answer for question N>",
      "reasoning": "<Reason for the answer based on the paragraph>",
      "location": "<Location in the paragraph (e.g., 'paragraph 4, line 2')>"
    }
  }
}

"""
        self.client = Together(base_url="https://api.aimlapi.com/v1", api_key=api_key)
        self.model = 'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo'
    
    def preprocess(self, input_string):
        # pdb.set_trace()
        output = input_string.choices[0].message.content.replace('\n', '').replace('json', '').replace('bash', '')
        output = re.sub(r'<br>', '\n', output)
        if '```' in output:
            start = output.find('```') + 3
            end = output.rfind('```')
            output = output[start:end].strip()
        
        elif '{' in output:
            start = output.find('{')
            end = output.rfind('}') + 1

            if start != -1 and end != -1:
                output = output[start:end]
            else:
                print("No JSON found in the text")
        
        output = output.replace("'", '"')
        print(output)
        
        # if type(output) == str:
        #     output = json.loads(output)
        return output
    
    def output_model(self, data, input_prompt):
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
                        "text":  input_prompt + data,
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


    def start_process(self):
        # Test feeding a specific local image to the Llama model
        image_path = 'C:/Users/manza/Documents/tesseract/images/p_1.png'
        question_path = 'C:/Users/manza/Documents/tesseract/images/q1.jpg'
        ans_path = 'C:/Users/manza/Documents/tesseract/images/a1.jpg'

        # output_para = self.preprocess(self.llama_model(image_path, self.user_prompt_paragraph))
        time.sleep(10)
        output_ques = self.preprocess(self.llama_model(question_path, self.user_prompt_questions))
        time.sleep(10)
        # output_ans = self.preprocess(self.llama_model(ans_path, self.user_prompt_solutions))

        # data = f"Paragraph: {str(output_para)} <===> Questions: {str(output_ques)} <===> Answers: {str(output_ans)}"
        pdb.set_trace()
        # final_output = self.preprocess(self.output_model(data, self.actual_prompt))
        

if __name__ == "__main__":
    product = HackathonProduct()
    product.start_process()
    
    # print(response12.choices[0].message.content)
