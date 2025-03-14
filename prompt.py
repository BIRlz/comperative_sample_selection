def system_prompt():
    return f"You are a useful Assistant to help select the optimal element. \
        You will receive two sets: Set A and Candidate Set B. Each element in both sets is a triplet [instruction, input, response]'. \
        Your objective is to identify the optimal element from Set B to add to Set A by following the criteria:\
        1. Response Quality: The response should be high-quality, relevant, coherent, and informative in relation to its instruction and input.\
        2. Marginal Contribution to Diversity: The element should maximize the diversity of the target set by introducing unique value."

def prompt_0(list_a, list_b):
    return f"### Example: \n\
    Set A: [Element_1, Element_2, …, Element_N] \n\
    Candidate Set B: [[A]-Element, [B]-Element, …, [N]-Element]\n\
    ###Steps: \n\
    1. Evaluate Response Quality: Assess the relevance, coherence, and informativeness of each element in Candidate Set B. \n\
    2. Add to Set A: Add each element from Candidate Set B to Set A to form new sets like: \n\
        - Adding [[A]-Element]  to Set A: [Element_1, Element_2, …, Element_N, [A]-Element] \n\
        - Adding [[B]-Element] to Set A: [Element_1, Element_2, …, Element_N, [B]-Element] \n\
        - … \n\
        - Adding [[N]-Element] to Set A: [Element_1, Element_2, …, Element_N, [N]-Element] \n\
    3. Assess Diversity Contribution and Select Optimal Element: Choose the Element from Candidate Set B that best improves Set A in terms of both response quality and diversity. \n\
    ### Here is the Input: \n\
    Set A: [{list_a}] \n\
    Candidate Set B: [{list_b}] \n\
    Finally, ONLY return the index of selected element from Set B by strictly following the format: [A] for the first one, [B] for the second one, etc. And in the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias..\n\
    ### Your Decision:"
    