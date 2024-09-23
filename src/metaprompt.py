import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class MetaPrompt:
    def __init__(self):
        # 获取当前脚本所在目录
        current_script_path = os.path.dirname(os.path.abspath(__file__))

        # 构建 metaprompt.txt 文件的完整路径
        prompt_guide_path = os.path.join(current_script_path, "metaprompt.txt")

        # 读取 metaprompt.txt 文件内容
        with open(prompt_guide_path, "r") as f:
            self.metaprompt = f.read()

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")  # 使用自定义的 API URL
        )

    def __call__(self, task, variables):
        variables = variables.split("\n")
        variables = [variable for variable in variables if len(variable)]

        variable_string = ""
        for variable in variables:
            variable_string += "\n{$" + variable.upper() + "}"
        prompt = self.metaprompt.replace("{{TASK}}", task)
        assistant_partial = "<Inputs>"
        if variable_string:
            assistant_partial += (
                variable_string + "\n</Inputs>\n<Instructions Structure>"
            )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_partial},
        ]
        print(f"openai.base_url: {self.client.base_url}\n openai.api_key: {self.client.api_key}")
        print(f"messages: {messages}")
        response = self.client.chat.completions.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=4096,
            temperature=0.0
        )
        message = response.choices[0].message.content

        extracted_prompt_template = self.extract_prompt(message)
        variables = self.extract_variables(message)

        return extracted_prompt_template.strip(), "\n".join(variables)

    def extract_between_tags(
        self, tag: str, string: str, strip: bool = False
    ) -> list[str]:
        ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
        if strip:
            ext_list = [e.strip() for e in ext_list]
        return ext_list

    def remove_empty_tags(self, text):
        return re.sub(r"\n<(\w+)>\s*</\1>\n", "", text, flags=re.DOTALL)

    def extract_prompt(self, metaprompt_response):
        between_tags = self.extract_between_tags("Instructions", metaprompt_response)[0]
        return (
            between_tags[:1000]
            + self.remove_empty_tags(
                self.remove_empty_tags(between_tags[1000:]).strip()
            ).strip()
        )

    def extract_variables(self, prompt):
        pattern = r"{([^}]+)}"
        variables = re.findall(pattern, prompt)
        return set(variables)

# 测试代码（如果需要的话）
# test = MetaPrompt()
# TASK = "Draft an email responding to a customer complaint"
# VARIABLES = ["CUSTOMER_COMPLAINT", "COMPANY_NAME"]
# test(TASK, VARIABLES)
