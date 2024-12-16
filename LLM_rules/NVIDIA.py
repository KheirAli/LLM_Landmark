from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)
state = "closeby(obj1,obj3), closeby(obj1,obj4), closeby(obj2,obj6), closeby(obj3,obj1), closeby(obj3,obj4), closeby(obj4,obj1), closeby(obj4,obj3), closeby(obj6,obj2), on_left(obj1,obj3), on_left(obj2,obj1), on_left(obj2,obj3), on_left(obj2,obj4), on_left(obj2,obj5), on_left(obj2,obj7), on_left(obj2,obj8), on_left(obj4,obj1), on_left(obj4,obj3), on_left(obj5,obj1), on_left(obj5,obj3), on_left(obj5,obj4), on_left(obj5,obj7), on_left(obj5,obj8), on_left(obj6,obj1), on_left(obj6,obj3), on_left(obj6,obj4), on_left(obj6,obj5), on_left(obj6,obj7), on_left(obj6,obj8), on_left(obj7,obj1), on_left(obj7,obj3), on_left(obj7,obj4), on_left(obj8,obj1), on_left(obj8,obj3), on_left(obj8,obj4), on_left(obj8,obj7), reach(obj2,obj6), reach(obj6,obj2), type(obj1,agent), type(obj3,door), type(obj4,enemy), type(obj5,coin), type(obj7,flag), type(obj8,key_red),  "
completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"The base rule was not sufficient and we reached this failing state: {state}\nYour task is to generate refined rules. In the first step, interpret every rule by examining the right-hand side of the rule and determine how we can make the rule more general by removing a predicate from the state predicates {on_left, type, closeby}. \nBase rules: \nright_to_coin(X) :- closeby(O1, O2), type(O1, agent), type(O6, coin), on_left(O6, O1). \nleft_to_coin(X) :- closeby(O1, O2), type(O1, agent), type(O6, coin), on_right(O6, O1). \nSuggested refined rules:"}],
  temperature=0.0,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

rule_pattern = re.compile(r"(\w+\(.*?\)\s*:-\s*.*?\.)", re.DOTALL)

rules = rule_pattern.findall(response_text)

print("Extracted Rules:")
for r in rules:
    # Clean up whitespaces and newlines
    rule_str = " ".join(r.split())
    print(rule_str)
