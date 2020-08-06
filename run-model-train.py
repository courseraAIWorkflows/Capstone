from model import model_train, model_load
import os
def main():
    
    ## train the model
    data_dir = os.path.join(".","data","cs-train")
    model_train(data_dir,test=False)

    ## load the model
    all_data,all_models = model_load()
    
    print("model training complete.")


if __name__ == "__main__":

    main()
