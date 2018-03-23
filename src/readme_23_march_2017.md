this code was backed on the 23rd of March/2018

The models were runs for 50 epochs each and the model was saved based on the basis of validation accuracy. However, the loss was stepped on basis of the loss.

The batch size was 4

The models was trained with both statistics and the model was saved on basis of the f1 score. 


v_model:
Folder comprising of various expert and primary models

primary_o and expert_o : All primary and expert models

primary and expert: These were model that where chosen by eye balling the accuracy. The accuracy attained on the validation data was used to cherry pick them

primay_a and expert_a: these were models that were picked by finding the model with highest f1 score and all models with f1 score atleast in the range of  95 %.


Based on results on test data, primary and expert gave a result of 82.14


+ 14 0 1 0 0
+ 2 10 0 0 0
+ 1 0 11 2 0
+ 0 0  1 8 0
+ 0 0 1  2 3


while using primary_a and expert_a gave:

+ 13 0 2 0 0
+ 2 10 0 0 0
+ 2 0 10 2 0
+ 0 0 1  8 0
+ 0 0 1  0 5 

Based on avinash comment we used primary and expert_a
+ 14 0 1 0 0
+ 2 10 0 0 0
+ 1 0 11 2 0
+ 0 0  1 8 0
+ 0 0 1  0 5 

