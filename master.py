import determine_labels
import identify_question
import sys

if len(sys.argv) != 2:
    print(None)
else:
    post = sys.argv[1]
    if identify_question.is_question(post):
        try:
            print(determine_labels.main(post))
        except:
            print(None)
    else:
        print(None)

