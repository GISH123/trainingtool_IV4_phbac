from models import get_model2idx_dict

def  show_model_list() :
    print("Models included: ")
    with open(file="models.txt", mode="w", encoding="utf-8") as f:
        for model_name, i in get_model2idx_dict().items():
            print("MODEL: {}, INDEX: {}".format(model_name, i))
            f.write("{}: {}\n".format(model_name, i))


if __name__ == '__main__':
     show_model_list()

