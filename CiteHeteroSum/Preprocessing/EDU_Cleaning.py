import re

def filter_invalid_edus(sent, edus):
    valid_edus = []

    for edu in edus:
        tokens = edu.split()
        num_tokens = len(tokens)
        has_long = any(len(token) > 30 for token in tokens)

        if num_tokens == 0:
            print(f"EDU has 0 tokens: '{edu}'\nBelongs to sentence: '{sent}'\n\n")
        elif num_tokens > 90:
            print(f"EDU has more than 90 tokens: '{edu}' - {num_tokens} token(s)\nBelongs to sentence: '{sent}'\n\n")
        elif has_long:
            print(f"EDU contains token with length > 30: '{edu}'\nBelongs to sentence: '{sent}'\n\n")
        else:
            valid_edus.append(edu)

    return valid_edus


def model_error_fix(sent, edus):
    token_dict = dict()
    for edu in edus:
        token_dict[edu] = edu.split()

    i = 0
    while i < len(edus):
        if len(token_dict[edus[i]]) < 4:
            if i > 0:
                previous_edu_tokens = token_dict[edus[ i -1]]
                last_previous_token = previous_edu_tokens[-1]
                merged_token = last_previous_token + edus[i]

                if merged_token in sent:
                    new_token = previous_edu_tokens[:-1] + [merged_token]
                    new_text = ' '.join(new_token)
                    token_dict[new_text] = new_token

                    edus[ i -1] = new_text
                    edus.pop(i)
                    continue

                if re.match(r'^[.,!?;:…)\]}]', edus[i]):
                    new_text = edus[ i -1] + ' ' + edus[i]
                    new_token = new_text.split()
                    token_dict[new_text] = new_token

                    edus[ i -1] = new_text
                    edus.pop(i)
                    continue

            if i < len(edus) - 1:
                next_edu_tokens = token_dict[edus[ i +1]]
                first_next_token = next_edu_tokens[0]
                merged_token = edus[i] + first_next_token

                if merged_token in sent:
                    new_token = [merged_token] + next_edu_tokens[1:]
                    new_text = ' '.join(new_token)
                    token_dict[new_text] = new_token
                    edus[ i +1] = new_text
                    edus.pop(i)
                    continue

        else:
            if i > 0:
                previous_edu_tokens = token_dict[edus[ i -1]]
                last_previous_token = previous_edu_tokens[-1]
                current_edu_tokens =  token_dict[edus[i]]
                first_current_token = current_edu_tokens[0]

                merged_token = last_previous_token + first_current_token

                if merged_token in sent:
                    new_token = current_edu_tokens[1:]
                    new_text = ' '.join(new_token)
                    edus[i] = new_text
                    token_dict[new_text] = new_token

                    new_token = previous_edu_tokens[:-1] + [merged_token]
                    new_text = ' '.join(new_token)
                    edus[i -1] = new_text
                    token_dict[new_text] = new_token

                if re.match(r'^[.,!?;:…)\]}]', first_current_token):
                    new_text = edus[ i -1] + " " + first_current_token
                    new_token = new_text.split()
                    token_dict[new_text] = new_token
                    edus[ i -1] = new_text

                    new_text = ' '.join(current_edu_tokens[1:])
                    new_token = new_text.split()
                    token_dict[new_text] = new_token
                    edus[i] = new_text

            if i < len(edus) - 1:
                current_edu_tokens = token_dict[edus[i]]
                last_current_token = current_edu_tokens[-1]
                next_edu_tokens = token_dict[edus[ i +1]]
                fist_next_token = next_edu_tokens[0]

                merged_token = last_current_token + fist_next_token

                if merged_token in sent:
                    new_token = next_edu_tokens[1:]
                    new_text = ' '.join(new_token)
                    token_dict[new_text] = new_token
                    edus[i +1] = new_text

                    new_token = current_edu_tokens[:-1] + [merged_token]
                    new_text = ' '.join(new_token)
                    token_dict[new_text] = new_token
                    edus[i] = new_text
        i += 1

    if len(edus) == 0 or edus[-1] in sent:
        pass
    else:
        total_previous_count = 0
        for edu in edus[:-1]:
            total_previous_count += len(token_dict[edu])

        new_token = sent.split()[total_previous_count:]
        new_text = ' '.join(new_token)
        token_dict[new_text] =  new_token
        edus[-1] = new_text
    return edus

def merge_edus(edus):
    i = 0
    while i<len(edus):
        if len(edus[i].split())<4:
            if i>0:
                edus[i-1] = edus[i-1] + " " + edus[i]
                edus.pop(i)
            elif i == 0 and i+1<len(edus):
                edus[i+1] = edus[i] + " " + edus[i+1]
                edus.pop(i)
        else:
            i += 1
    return edus

if __name__ == '__main__':
    sent = "the nebula was mapped in the '' on - the - fly '' mode , in which data are collected while the telescope scans in a raster pattern ."
    edus = ["the nebula was mapped in the '' on - the - fly '' mode", ', in which data are collect', 'ed while the telescope scans in a raster pattern .']
    edus = filter_invalid_edus(sent, edus)
    edus = model_error_fix(sent, edus)
    edus = merge_edus(edus)
    for edu in edus:
        print(edu)

