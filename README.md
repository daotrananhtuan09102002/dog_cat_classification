# Exercise 1: Image Classification

Download the notebook `demo_and_evaluation.ipynb` and upload it to `Google Colab` environment.

1.1 Inferece on a single image in web browser

- Run the first section to import the necessary libraries and clone the repository.

- Run the second section (except the last cell) to deploy the model and start the web server.

- Click on the link to open the web page.

- Click on the `Choose File` button to select an image from your computer or drag and drop an image (from your computer or Internet) into the box. 

- Click on the `Classify` button to run the inference and display the result.

- When you are done, close the web page and go back to the notebook, run the last cell to stop the web server.

1.2 Evaluation on the test set

- Run the last section of the notebook to evaluate the model on the test set.

- Your test set can be uploaded to `Google Drive` and mounted to the notebook or uploaded directly to the `Google Colab` environment.

- The test set has structure like this:

```bash
test
├── cat
│   ├── image_0.jpg
│   ├── image_1.jpg
│   ├── ... 
├── dog
│   ├── image_0.jpg
│   ├── image_1.jpg
│   ├── ...

```

- Run command `python utils.py --model_path <path_to_model.h5> --dataset_path <path_to_test_set> --batch_size <batch_size>`

- The accuracy on the test set will be printed out.


