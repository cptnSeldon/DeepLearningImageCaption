""" ----------------
    DATA PREPARATION : may take a while
    ---------------- """
# https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
import string

def load_doc(filename):
  """
  param:  filename
  return: text

  This function does the following:
    - retrives all the text contained in the descriptions' file
    - returns the retrieved (loading into the memory)
  """
    file = open(filename, 'r')
    text = file.read()
    file.close()

    return text

def load_descriptions(doc):
    """
    param:  doc (= text returned by the function above)
    return: mapping

    This function does the following:
      - creates a dictionary which will contain
        - the image identifier
        - the image description
      - retrieve the id and the description (parsing)
        - line composition : [image_id] space [description] new line
        - dictionary: mapping[image_id] = [description_1, description_2, ..]
      - returns the resulting dictionary
    """
    mapping = dict()

    for line in doc.split():

        tokens = line.split()

        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)

        if image_id not in mapping:
            mapping[image_id] = list()

        mapping[image_id].append(image_desc)

    return mapping

def clean_descriptions(descriptions):
    """
    param:  descriptions
    return: none

    This function cleans the unnecessary information:
    - the sentence is cut into words (tokenization) then converted to lowercase
    - the punctuation is removed as well as the remaining a and s
    (we could also remove the other determinants here..)
    - the numbers are removed too
    """
    table = str.maketrans('', '', string.punctuation)

    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

def to_vocabulary(descriptions):
    """
    param:  descriptions
    return: all descriptions

    Creates a list containing all the descriptions in string.
    """
    all_desc = set()

    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]

    return all_desc


def save_descriptions(descriptions, filename):
    """
    param:  descriptions, filename
    return: none

    Saves the descriptions' list in the given file.
    """
    lines = list()

    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)

    data = '\n'.join(lines)

    file = open(filename, 'w')
    file.write(data)
    file.close()


if __name__ == "__main__":

    filename = '../Flickr8k_text/Flickr8k.token.txt'

    doc = load_doc(filename)    # load descriptions
    descriptions = load_descriptions(doc)   # parse them
    print('Loaded: %d ' % len(descriptions))

    clean_descriptions(descriptions)    # clean them

    vocabulary = to_vocabulary(descriptions)    # summarize vocabulary
    print('Vocabulary size: %d' % len(vocabulary))

    save_descriptions(descriptions, 'descriptions.txt') # save descriptions to file
