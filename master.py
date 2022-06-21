import determine_labels
import identify_question

def response(post):
    if identify_question.is_question(post):
        try:
            return determine_labels.main(post)
        except:
            return None
    return None

