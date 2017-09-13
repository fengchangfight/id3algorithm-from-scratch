"""
==fc== ID3 algorithm, build decision tree
"""
import math

# ==fc== Calculates the entropy of the given data set for the target attribute.
def entropy(data, target_attr):
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (record[target_attr] in val_freq):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)

    return data_entropy

# ==fc== calc information gain
def gain(data, attr, target_attr):
    val_freq = {}
    subset_entropy = 0.0
    for record in data:
        if (record[attr] in val_freq):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)
    return (entropy(data, target_attr) - subset_entropy)

#==fc== this is recursive method, which always choose the best attr and subdivide the dataset into sub dataset to make subtrees until no node
def create_decision_tree(parentdata, data, attributes, target_attr, possible_dic, fitness_func):
    """
    Returns a new decision tree based on the examples given.
    """
    data = data[:]
    vals = [record[target_attr] for record in data]
    if not data :
        return majority_value(parentdata, target_attr)
    elif len(attributes) <= 1:
        return majority_value(data, target_attr)
    #==fc== all the same values
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # ==fc== Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr,fitness_func)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best: {}}

        # ==fc== create sub tree
        for val in possible_dic[best]:
            # ==fc== create sub tree
            subtree = create_decision_tree(
                data,
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr, possible_dic,
                fitness_func)

            tree[best][val] = subtree

    return tree


########################## Helper Functions #####################


# ==fc== get the majority target value across the data set
def majority_value(data, target_attr):
    data = data[:]
    return most_frequent([record[target_attr] for record in data])

# ==fc== return the most frequently occured value
def most_frequent(lst):
    lst = lst[:]
    highest_freq = 0
    most_freq = None
    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
    return most_freq

# ==fc== remove duplicate
def unique(lst):
    lst = lst[:]
    unique_lst = []
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
    return unique_lst

# ==fc== remove duplicate attribute values
def get_values(data, attr):
    data = data[:]
    return unique([record[attr] for record in data])

# ==fc== choose the best attribute by best information gain, fitness is a helper function passed in
def choose_attribute(data, attributes, target_attr, fitness):
    data = data[:]
    best_gain = 0.0
    best_attr = None
    for attr in attributes:
        gain = fitness(data, attr, target_attr)
        if (gain >= best_gain and attr != target_attr):
            best_gain = gain
            best_attr = attr
    return best_attr

# ==fc== get data has attr of specific value
def get_examples(data, attr, value):
    data = data[:]
    rtn_lst = []

    if not data:
        return rtn_lst
    else:
        for record in data:
            if record[attr] == value:
                rtn_lst.append(record)

        return rtn_lst

#==fc== this is a recursive function to get classification of a data record
def get_classification(record, tree):
    if isinstance(tree, str):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = list(tree.keys())[0]
        t = tree[attr][record[attr]]
        return get_classification(record, t)

#==fc== return a list of classifications of data
def classify(tree, data):
    data = data[:]
    classification = []

    for record in data:
        classification.append(get_classification(record, tree))

    return classification