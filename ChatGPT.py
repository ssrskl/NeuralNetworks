import openai
openai.api_key="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"
# 代码实现gpt对文本进行总结
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # 设置生成文本的随机程度
    )
    return response.choices[0].message["content"]

# 策略1   使用分隔符
# '''{text}'''三引号是分隔符，可以让gpt清晰的划分文本
#
text = f"""
You should express what you want a model to do by \
providing instructions that are as clear and \
specific as you can possibly make them. \
This will guide the model towards the desired output,\
and reduce the chances of receiving irrelevant \
or incorrect responses. Don't confuse writing a \
clear prompt with writing a short prompt. \
In many cases, longer prompts provide more clarity \
and context for the model, which can lead to \
more detailed and relevant outputs.''
"""
prompt =f"""
Summarize the text delimited by triple backticks \
into a single sentence.
'''{text}'''
"""
response = get_completion(prompt)
print (response)