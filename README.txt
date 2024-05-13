Load the Data
It classifies the numbers in the document so we know which is which , e.x  0 is
the first 200 examples , then 1 is the next 200 and so on.


Preprocess the Data

We make sure that all the numbers (which were written in a special kind of ink
that can look heavier or lighter) are easier to compare by making the ink levels
consistent. We also rearrange each drawing into a specific shape and size that
fits into our robot's scanner perfectly. Finally, we make the tags easier for the
robot to read by changing them into a special code.


Split the Data
Splits the data one for training and one for practising .


Build the CNN model

Conv2D layers: These are like eyes that help the robot focus on small parts of the drawing to notice patterns and details.
MaxPooling2D layers: These make the pictures smaller so the robot can manage them more easily without getting overwhelmed.
Flatten: This turns the picture into a long string of numbers that the robot can think about all at once.
Dense: These are decision-making layers where the robot uses what it's seen to make guesses about what number is being shown.
Dropout: Sometimes we make the robot forget some of what it's seen to make sure it isn't just memorizing the pictures.


Train the Model

Now we take all the training pictures and let the robot practice over and over
(for as many times as we set in 'epochs') to get really good at recognizing the
numbers. It tries to guess each one, and then we tell it if it was right or wrong
so it can learn.


Evaluate the Model


Finally, we take the test pile of pictures and see how well the robot can guess them without any help. We look at how many it got right (accuracy) and how many it missed (error rate). It's like giving the robot a final exam to see how much it has learned from all its practice!

This whole process helps the robot learn to recognize numbers from drawings, just like you learn to recognize letters and numbers when you practice reading!
