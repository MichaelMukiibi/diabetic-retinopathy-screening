from fastdrs.training import train_model

if __name__ == "__main__":
    train_model(
        architecture="resnet50",
        epochs=5,
    )