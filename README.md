About this competition
---------------
In this competition, we are asked to build a model that classifies hand-drawn images. The model you build will classify 340 labels, and your model will be ranked based on top-3 accuracy of your model prediction.


Result
---------------
Individual models gave me the following result.

| Models            | Use Imagenet | Public LB  | Private LB  |
| ----------------- |:------------:|:----------:|:-----------:|
| Inception Resnet  |       O      | 0.92898    | 0.93034     |
| Inception Resnet  |       X      | 0.93123    | 0.92961     |
| Xception          |       O      | 0.91699    | 0.91851     |
| Xception          |       X      | 0.93040    | 0.92964     |

When averaging the above model, our final result was 0.94034 which is 86th out of 1316 participants.
Note that this is unofficial result due to the late submission. For anyone who want to read more about this competiotion, you can visit my [**github page**](https://jinkilee.github.io/jekyll/update/doodle_classification/)
