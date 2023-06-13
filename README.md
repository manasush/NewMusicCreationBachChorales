# New Music Creation From Bach Chorales
Using the dataset containing 382 chorales by Johann Sebastian Bach and generating new music




Dataset contains 382 chorales by Johann Sebastian Bach (in the public domain), where each chorale is composed of 100 to 640 chords with a temporal resolution of 1/16th (ie each chorale is 100 to 640 sixteenth-notes long). Each chord has four notes, composed of 4 integers, each indicating the index of a note on a piano, except for the value 0 which means "no note played."

To get longer notes, chords are repeated (ie, if a chord is repeated twice it is equivalent to an 1/8 note, repeated four times it is equivalent to a quarter note, etc). Note that this doesn't truly represent music as in music you can repeat a note and it means to replay it, not to lengthen its duration. But, this is where we start for this project.




# Applied Neural Networks
**CS 4499/5599**

# Undergraduate Final Project
To create new music from the Bach Chorales <br/>
Instructor: Dr.Leslie Kerby
<Br>
T/A: Samantha Ross<Br>
Sushan Manandhar<Br>
Date: May 5 2022<Br>

**Citation: Most of the code has been retrived from the file provided on Moodle and Google Colab (RNN) <Br>
  
  
  This dataset contains 382 chorales by Johann Sebastian Bach (in the public domain), where each chorale is composed of 100 to 640 chords with a temporal resolution of 1/16th (ie each chorale is 100 to 640 sixteenth-notes long). Each chord has four notes, composed of 4 integers, each indicating the index of a note on a piano, except for the value 0 which means "no note played."

To get longer notes, chords are repeated (ie, if a chord is repeated twice it is equivalent to an 1/8 note, repeated four times it is equivalent to a quarter note, etc). Note that this doesn't truly represent music as in music you can repeat a note and it means to replay it, not to lengthen its duration. But, this is where we start for this project.
  
  
  **Part 0.1**</br>
Set up the chorale dataset. I do this for you. If you want to change where the chorales are stored etc, feel free.
  
  
  
  
  ```python:
  # Standard imports
# Change runtime to GPU!
import tensorflow as tf
from tensorflow import keras
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)
  
  
  
  
  ```
  Downloading data from https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz
122880/117661 [===============================] - 0s 0us/step
131072/117661 [=================================] - 0s 0us/step
  
  
  
   ```python:
  filepath
  ```
  '/root/.keras/datasets/bach/bach.tgz'

  
  ```python:
  # Notice the chorales have been extracted into folders for train, test, and test
! ls /root/.keras/datasets/bach/
  ```
  bach.tgz  test	train  valid
 ```python:
  # 229 chorales in the training set, 76 in the validation set, and 77 in testing
! ls /root/.keras/datasets/bach/train/
  ```
  Downloading data from https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz
122880/117661 [===============================] - 0s 0us/step
131072/117661 [=================================] - 0s 0us/step
'/root/.keras/datasets/bach/bach.tgz'
bach.tgz  test	train  valid
chorale_000.csv  chorale_058.csv  chorale_116.csv  chorale_174.csv
chorale_001.csv  chorale_059.csv  chorale_117.csv  chorale_175.csv
chorale_002.csv  chorale_060.csv  chorale_118.csv  chorale_176.csv
chorale_003.csv  chorale_061.csv  chorale_119.csv  chorale_177.csv
chorale_004.csv  chorale_062.csv  chorale_120.csv  chorale_178.csv
chorale_005.csv  chorale_063.csv  chorale_121.csv  chorale_179.csv
chorale_006.csv  chorale_064.csv  chorale_122.csv  chorale_180.csv
chorale_007.csv  chorale_065.csv  chorale_123.csv  chorale_181.csv
chorale_008.csv  chorale_066.csv  chorale_124.csv  chorale_182.csv
chorale_009.csv  chorale_067.csv  chorale_125.csv  chorale_183.csv
chorale_010.csv  chorale_068.csv  chorale_126.csv  chorale_184.csv
chorale_011.csv  chorale_069.csv  chorale_127.csv  chorale_185.csv
chorale_012.csv  chorale_070.csv  chorale_128.csv  chorale_186.csv
chorale_013.csv  chorale_071.csv  chorale_129.csv  chorale_187.csv
chorale_014.csv  chorale_072.csv  chorale_130.csv  chorale_188.csv
chorale_015.csv  chorale_073.csv  chorale_131.csv  chorale_189.csv
chorale_016.csv  chorale_074.csv  chorale_132.csv  chorale_190.csv
chorale_017.csv  chorale_075.csv  chorale_133.csv  chorale_191.csv
chorale_018.csv  chorale_076.csv  chorale_134.csv  chorale_192.csv
chorale_019.csv  chorale_077.csv  chorale_135.csv  chorale_193.csv
chorale_020.csv  chorale_078.csv  chorale_136.csv  chorale_194.csv
chorale_021.csv  chorale_079.csv  chorale_137.csv  chorale_195.csv
chorale_022.csv  chorale_080.csv  chorale_138.csv  chorale_196.csv
chorale_023.csv  chorale_081.csv  chorale_139.csv  chorale_197.csv
chorale_024.csv  chorale_082.csv  chorale_140.csv  chorale_198.csv
...
chorale_054.csv  chorale_112.csv  chorale_170.csv  chorale_228.csv
chorale_055.csv  chorale_113.csv  chorale_171.csv
chorale_056.csv  chorale_114.csv  chorale_172.csv
chorale_057.csv  chorale_115.csv  chorale_173.csv
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
[PosixPath('/root/.keras/datasets/bach/train/chorale_000.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_001.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_002.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_003.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_004.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_005.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_006.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_007.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_008.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_009.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_010.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_011.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_012.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_013.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_014.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_015.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_016.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_017.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_018.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_019.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_020.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_021.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_022.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_023.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_024.csv'),
...
 PosixPath('/root/.keras/datasets/bach/train/chorale_224.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_225.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_226.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_227.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_228.csv')]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
  
  ```python:
  from pathlib import Path
jsb_chorales_dir = Path(filepath).parent
train_files = sorted(jsb_chorales_dir.glob("train/*.*"))
valid_files = sorted(jsb_chorales_dir.glob("valid/*.*"))
test_files = sorted(jsb_chorales_dir.glob("test/*.*"))
train_files
# You don't have to have sorted (would be a generator without it)
  ```
  [PosixPath('/root/.keras/datasets/bach/train/chorale_000.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_001.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_002.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_003.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_004.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_005.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_006.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_007.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_008.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_009.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_010.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_011.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_012.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_013.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_014.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_015.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_016.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_017.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_018.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_019.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_020.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_021.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_022.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_023.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_024.csv'),
...
 PosixPath('/root/.keras/datasets/bach/train/chorale_224.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_225.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_226.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_227.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_228.csv')]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
  
 ```python:  
  import pandas as pd

# Creates a list of ndarray-chorales
def load_chorales(filepaths):
    return [pd.read_csv(filepath).values for filepath in filepaths]

train = np.array(load_chorales(train_files))
valid = np.array(load_chorales(valid_files))
test = np.array(load_chorales(test_files))
  ````
  /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:7: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  import sys
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:8: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:9: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  if __name__ == '__main__':
  
  
  
  ```python:  
  train[0]
  ````
  
  Downloading data from https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz
122880/117661 [===============================] - 0s 0us/step
131072/117661 [=================================] - 0s 0us/step
'/root/.keras/datasets/bach/bach.tgz'
bach.tgz  test	train  valid
chorale_000.csv  chorale_058.csv  chorale_116.csv  chorale_174.csv
chorale_001.csv  chorale_059.csv  chorale_117.csv  chorale_175.csv
chorale_002.csv  chorale_060.csv  chorale_118.csv  chorale_176.csv
chorale_003.csv  chorale_061.csv  chorale_119.csv  chorale_177.csv
chorale_004.csv  chorale_062.csv  chorale_120.csv  chorale_178.csv
chorale_005.csv  chorale_063.csv  chorale_121.csv  chorale_179.csv
chorale_006.csv  chorale_064.csv  chorale_122.csv  chorale_180.csv
chorale_007.csv  chorale_065.csv  chorale_123.csv  chorale_181.csv
chorale_008.csv  chorale_066.csv  chorale_124.csv  chorale_182.csv
chorale_009.csv  chorale_067.csv  chorale_125.csv  chorale_183.csv
chorale_010.csv  chorale_068.csv  chorale_126.csv  chorale_184.csv
chorale_011.csv  chorale_069.csv  chorale_127.csv  chorale_185.csv
chorale_012.csv  chorale_070.csv  chorale_128.csv  chorale_186.csv
chorale_013.csv  chorale_071.csv  chorale_129.csv  chorale_187.csv
chorale_014.csv  chorale_072.csv  chorale_130.csv  chorale_188.csv
chorale_015.csv  chorale_073.csv  chorale_131.csv  chorale_189.csv
chorale_016.csv  chorale_074.csv  chorale_132.csv  chorale_190.csv
chorale_017.csv  chorale_075.csv  chorale_133.csv  chorale_191.csv
chorale_018.csv  chorale_076.csv  chorale_134.csv  chorale_192.csv
chorale_019.csv  chorale_077.csv  chorale_135.csv  chorale_193.csv
chorale_020.csv  chorale_078.csv  chorale_136.csv  chorale_194.csv
chorale_021.csv  chorale_079.csv  chorale_137.csv  chorale_195.csv
chorale_022.csv  chorale_080.csv  chorale_138.csv  chorale_196.csv
chorale_023.csv  chorale_081.csv  chorale_139.csv  chorale_197.csv
chorale_024.csv  chorale_082.csv  chorale_140.csv  chorale_198.csv
...
chorale_054.csv  chorale_112.csv  chorale_170.csv  chorale_228.csv
chorale_055.csv  chorale_113.csv  chorale_171.csv
chorale_056.csv  chorale_114.csv  chorale_172.csv
chorale_057.csv  chorale_115.csv  chorale_173.csv
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
[PosixPath('/root/.keras/datasets/bach/train/chorale_000.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_001.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_002.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_003.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_004.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_005.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_006.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_007.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_008.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_009.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_010.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_011.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_012.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_013.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_014.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_015.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_016.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_017.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_018.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_019.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_020.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_021.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_022.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_023.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_024.csv'),
...
 PosixPath('/root/.keras/datasets/bach/train/chorale_224.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_225.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_226.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_227.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_228.csv')]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:7: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  import sys
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:8: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:9: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  if __name__ == '__main__':
array([[74, 70, 65, 58],
       [74, 70, 65, 58],
       [74, 70, 65, 58],
       [74, 70, 65, 58],
       [75, 70, 58, 55],
       [75, 70, 58, 55],
       [75, 70, 60, 55],
       [75, 70, 60, 55],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 70, 62, 55],
       [77, 70, 62, 55],
       [77, 69, 62, 55],
       [77, 69, 62, 55],
       [75, 67, 63, 48],
       [75, 67, 63, 48],
       [75, 69, 63, 48],
       [75, 69, 63, 48],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [72, 69, 65, 53],
...
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46]])
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
(229,)
(192, 4)
(229,)
' 74706558 74706558 74706558 74706558 75705855 75705855 75706055 75706055 77696250 77696250 77696250 77696250 77706255 77706255 77696255 77696255 75676348 75676348 75696348 75696348 74706546 74706546 74706546 74706546 72696553 72696553 72696553 72696553 72696553 72696553 72696553 72696553 74706546 74706546 74706546 74706546 75696348 75696348 75676348 75676348 77656250 77656250 77656050 77656050 74675855 74675855 74675853 74675853 72675851 72675851 72675851 72675851 72655753 72655753 72655753 72655753 70656246 70656246 70656246 70656246 70656246 70656246 70656246 70656246 72696553 72696553 72696553 72696553 74715350 74715350 74715350 74715350 75725548 75725548 75725550 75725550 75676051 75676051 75676053 75676053 74676055 74676055 74675755 74675755 74655943 74655943 72635943 72635943 72635548 72635548 72635548 72635548 72635548 72635548 72635548 72635548 75676060 75676060 75676060 75676060 77706258 77706258 77706256 77706256 79706255 79706255 79706253 79706253 79706351 79706351 79706351 79706351 77706358 77706358 77706058 77706058 77706246 77706246 77686246 75686246 75675851 75675851 75675851 75675851 75675851 75675851 75675851 75675851 74675855 74675855 74675855 74675855 75675853 75675853 75675851 75675851 77655850 77655850 77655650 77655650 70635551 70635551 70635551 70635551 75656045 75656045 75656045 75656045 74655846 74655846 74655846 74655846 72655753 72655753 72655753 72655753 72655753 72655753 72655753 72655753 74655858 74655858 74655858 74655858 75675857 75675857 75675855 75675855 77656057 77656057 77656053 77656053 74655858 74655858 74655858 74655858 72675851 72675851 72675851 72675851 72655753 72655753 72655753 72655753 70656246 70656246 70656246 70656246 70656246 70656246 70656246 70656246'
4419
array([ 102,  102,  102,  102,  862,  862, 1303, 1303,  425,  425,  425,
        425,  579,  579,  580,  580,  475,  475,  476,  476,   44,   44,
         44,   44,   36,   36,   36,   36,   36,   36,   36,   36,   44,
         44,   44,   44,  476,  476,  475,  475,  338,  338, 1023, 1023,
         87,   87,  581,  581,  134,  134,  134,  134,   12,   12,   12,
         12,   17,   17,   17,   17,   17,   17,   17,   17,   36,   36,
         36,   36, 1679, 1679, 1679, 1679, 2521, 2521, 2522, 2522,  426,
        426, 1024, 1024,  393,  393, 1266, 1266,  465,  465, 1680, 1680,
        169,  169,  169,  169,  169,  169,  169,  169,  339,  339,  339,
        339,  340,  340, 1681, 1681, 1682, 1682, 2523, 2523,  279,  279,
        279,  279, 2524, 2524, 2525, 2525,  466,  466, 1304, 2526,   76,
         76,   76,   76,   76,   76,   76,   76,   87,   87,   87,   87,
        863,  863,   76,   76,  582,  582, 2527, 2527,  230,  230,  230,
        230,  318,  318,  318,  318,   63,   63,   63,   63,   12,   12,
         12,   12,   12,   12,   12,   12,   52,   52,   52,   52, 1683,
       1683,  239,  239,  319,  319, 2528, 2528,   52,   52,   52,   52,
        134,  134,  134,  134,   12,   12,   12,   12,   17,   17,   17,
         17,   17,   17,   17,   17])
{'67625943': 1,
 '69666250': 2,
 '71676255': 3,
 '71686452': 4,
 '69646145': 5,
 '69656053': 6,
 '74665750': 7,
 '66625750': 8,
 '67646048': 9,
 '65605741': 10,
 '70676255': 11,
 '69646045': 12,
 '72655753': 13,
 '67645952': 14,
 '67625843': 15,
 '74675955': 16,
 '64595652': 17,
 '70656246': 18,
 '69625450': 19,
 '72645548': 20,
 '65625750': 21,
 '73696457': 22,
 '71645652': 23,
 '72676448': 24,
 '64615745': 25,
...
 '73645756': 997,
 '71625647': 998,
 "69615245'": 999,
 '64625747': 1000,
 ...}
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
4419
['72696551']
array([[949]])
  
  
  
  ```python:
  train.shape
# Why does this just show 1D?
#    --> Because it is a nested variable-length ndarray/list (ie like a ragged tensor)

```
  (229,)

  
  ```python:
train[0].shape # Each chorale has a different number of chords in its sequence
# Which works fine with RNNs :)
# Notes range from 36 (C1 = C on octave 1) to 81 (A5 = A on octave 5), plus 0 for silence.
# A on octave 4 (used for tuning) is note 69.
```

  (192, 4)

  
  ```python:
# Functions to play a sequence of chords; use them as you'd like, or create your own
# All we need to do is call play_chords(chord_sequence) -- see next cell
from IPython.display import Audio

def notes_to_frequencies(notes):
    # Frequency doubles when you go up one octave; there are 12 semi-tones
    # per octave; Note A on octave 4 is 440 Hz, and it is note number 69.
    return 2 ** ((np.array(notes) - 69) / 12) * 440

def frequencies_to_samples(frequencies, tempo, sample_rate):
    note_duration = 60 / tempo # the tempo is measured in beats per minutes
    # To reduce click sound at every beat, we round the frequencies to try to
    # get the samples close to zero at the end of each note.
    frequencies = np.round(note_duration * frequencies) / note_duration
    n_samples = int(note_duration * sample_rate)
    time = np.linspace(0, note_duration, n_samples)
    sine_waves = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
    # Removing all notes with frequencies ≤ 9 Hz (includes note 0 = silence)
    sine_waves *= (frequencies > 9.).reshape(-1, 1)
    return sine_waves.reshape(-1)

def chords_to_samples(chords, tempo, sample_rate):
    freqs = notes_to_frequencies(chords)
    freqs = np.r_[freqs, freqs[-1:]] # make last note a bit longer
    merged = np.mean([frequencies_to_samples(melody, tempo, sample_rate)
                     for melody in freqs.T], axis=0)
    n_fade_out_samples = sample_rate * 60 // tempo # fade out last note
    fade_out = np.linspace(1., 0., n_fade_out_samples)**2
    merged[-n_fade_out_samples:] *= fade_out
    return merged

def play_chords(chords, tempo=160, amplitude=0.1, sample_rate=44100, filepath=None):
    samples = amplitude * chords_to_samples(chords, tempo, sample_rate)
    if filepath:
        from scipy.io import wavfile
        samples = (2**15 * samples).astype(np.int16)
        wavfile.write(filepath, sample_rate, samples)
        return display(Audio(filepath))
    else:
        return display(Audio(samples, rate=sample_rate))
```

  
  
  ```python:
# First 3 chorales in training set
for index in range(3):
    play_chords(train[index])
```

  **Part 0.2**</br>
Create a sequence-to-sequence RNN that can predict the next chord (4 notes).

Use 64 chords to train (ie have windows of 64+1, and need to have data in tf.data.Dataset). Use `Embeddings` to encode each chord (instead of passing note-integers to the RNN).
  
  ```python:
# make deep copy of train 
import copy
train_copy = copy.deepcopy(train)
train_copy.shape
```

  (229,)

  
  ```python:
# Convert each chorale to a string with 8-digit integer IDs for chords

for i in range(len(train)): # iterate through each chorale in train
  str_chorale = ""
  for j in range(len(train[i])): # iterate through each chord in chorale i
    str_chord = ""
    for k in range(4):  # iterate through 4 notes (k) in chord j of chorale i
      str_note = str(train[i][j][k])
      str_chord = str_chord + str_note
      #print(str_note, str_chord, train[i][j])
    str_chorale = str_chorale + " " + str_chord
  train_copy[i] = str_chorale
```

  
  
  ```python:
train_copy[0]
```

  Downloading data from https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz
122880/117661 [===============================] - 0s 0us/step
131072/117661 [=================================] - 0s 0us/step
'/root/.keras/datasets/bach/bach.tgz'
bach.tgz  test	train  valid
chorale_000.csv  chorale_058.csv  chorale_116.csv  chorale_174.csv
chorale_001.csv  chorale_059.csv  chorale_117.csv  chorale_175.csv
chorale_002.csv  chorale_060.csv  chorale_118.csv  chorale_176.csv
chorale_003.csv  chorale_061.csv  chorale_119.csv  chorale_177.csv
chorale_004.csv  chorale_062.csv  chorale_120.csv  chorale_178.csv
chorale_005.csv  chorale_063.csv  chorale_121.csv  chorale_179.csv
chorale_006.csv  chorale_064.csv  chorale_122.csv  chorale_180.csv
chorale_007.csv  chorale_065.csv  chorale_123.csv  chorale_181.csv
chorale_008.csv  chorale_066.csv  chorale_124.csv  chorale_182.csv
chorale_009.csv  chorale_067.csv  chorale_125.csv  chorale_183.csv
chorale_010.csv  chorale_068.csv  chorale_126.csv  chorale_184.csv
chorale_011.csv  chorale_069.csv  chorale_127.csv  chorale_185.csv
chorale_012.csv  chorale_070.csv  chorale_128.csv  chorale_186.csv
chorale_013.csv  chorale_071.csv  chorale_129.csv  chorale_187.csv
chorale_014.csv  chorale_072.csv  chorale_130.csv  chorale_188.csv
chorale_015.csv  chorale_073.csv  chorale_131.csv  chorale_189.csv
chorale_016.csv  chorale_074.csv  chorale_132.csv  chorale_190.csv
chorale_017.csv  chorale_075.csv  chorale_133.csv  chorale_191.csv
chorale_018.csv  chorale_076.csv  chorale_134.csv  chorale_192.csv
chorale_019.csv  chorale_077.csv  chorale_135.csv  chorale_193.csv
chorale_020.csv  chorale_078.csv  chorale_136.csv  chorale_194.csv
chorale_021.csv  chorale_079.csv  chorale_137.csv  chorale_195.csv
chorale_022.csv  chorale_080.csv  chorale_138.csv  chorale_196.csv
chorale_023.csv  chorale_081.csv  chorale_139.csv  chorale_197.csv
chorale_024.csv  chorale_082.csv  chorale_140.csv  chorale_198.csv
...
chorale_054.csv  chorale_112.csv  chorale_170.csv  chorale_228.csv
chorale_055.csv  chorale_113.csv  chorale_171.csv
chorale_056.csv  chorale_114.csv  chorale_172.csv
chorale_057.csv  chorale_115.csv  chorale_173.csv
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
[PosixPath('/root/.keras/datasets/bach/train/chorale_000.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_001.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_002.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_003.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_004.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_005.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_006.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_007.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_008.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_009.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_010.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_011.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_012.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_013.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_014.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_015.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_016.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_017.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_018.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_019.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_020.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_021.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_022.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_023.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_024.csv'),
...
 PosixPath('/root/.keras/datasets/bach/train/chorale_224.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_225.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_226.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_227.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_228.csv')]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:7: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  import sys
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:8: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:9: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  if __name__ == '__main__':
array([[74, 70, 65, 58],
       [74, 70, 65, 58],
       [74, 70, 65, 58],
       [74, 70, 65, 58],
       [75, 70, 58, 55],
       [75, 70, 58, 55],
       [75, 70, 60, 55],
       [75, 70, 60, 55],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 70, 62, 55],
       [77, 70, 62, 55],
       [77, 69, 62, 55],
       [77, 69, 62, 55],
       [75, 67, 63, 48],
       [75, 67, 63, 48],
       [75, 69, 63, 48],
       [75, 69, 63, 48],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [72, 69, 65, 53],
...
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46]])
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
(229,)
(192, 4)
(229,)
' 74706558 74706558 74706558 74706558 75705855 75705855 75706055 75706055 77696250 77696250 77696250 77696250 77706255 77706255 77696255 77696255 75676348 75676348 75696348 75696348 74706546 74706546 74706546 74706546 72696553 72696553 72696553 72696553 72696553 72696553 72696553 72696553 74706546 74706546 74706546 74706546 75696348 75696348 75676348 75676348 77656250 77656250 77656050 77656050 74675855 74675855 74675853 74675853 72675851 72675851 72675851 72675851 72655753 72655753 72655753 72655753 70656246 70656246 70656246 70656246 70656246 70656246 70656246 70656246 72696553 72696553 72696553 72696553 74715350 74715350 74715350 74715350 75725548 75725548 75725550 75725550 75676051 75676051 75676053 75676053 74676055 74676055 74675755 74675755 74655943 74655943 72635943 72635943 72635548 72635548 72635548 72635548 72635548 72635548 72635548 72635548 75676060 75676060 75676060 75676060 77706258 77706258 77706256 77706256 79706255 79706255 79706253 79706253 79706351 79706351 79706351 79706351 77706358 77706358 77706058 77706058 77706246 77706246 77686246 75686246 75675851 75675851 75675851 75675851 75675851 75675851 75675851 75675851 74675855 74675855 74675855 74675855 75675853 75675853 75675851 75675851 77655850 77655850 77655650 77655650 70635551 70635551 70635551 70635551 75656045 75656045 75656045 75656045 74655846 74655846 74655846 74655846 72655753 72655753 72655753 72655753 72655753 72655753 72655753 72655753 74655858 74655858 74655858 74655858 75675857 75675857 75675855 75675855 77656057 77656057 77656053 77656053 74655858 74655858 74655858 74655858 72675851 72675851 72675851 72675851 72655753 72655753 72655753 72655753 70656246 70656246 70656246 70656246 70656246 70656246 70656246 70656246'
4419
array([ 102,  102,  102,  102,  862,  862, 1303, 1303,  425,  425,  425,
        425,  579,  579,  580,  580,  475,  475,  476,  476,   44,   44,
         44,   44,   36,   36,   36,   36,   36,   36,   36,   36,   44,
         44,   44,   44,  476,  476,  475,  475,  338,  338, 1023, 1023,
         87,   87,  581,  581,  134,  134,  134,  134,   12,   12,   12,
         12,   17,   17,   17,   17,   17,   17,   17,   17,   36,   36,
         36,   36, 1679, 1679, 1679, 1679, 2521, 2521, 2522, 2522,  426,
        426, 1024, 1024,  393,  393, 1266, 1266,  465,  465, 1680, 1680,
        169,  169,  169,  169,  169,  169,  169,  169,  339,  339,  339,
        339,  340,  340, 1681, 1681, 1682, 1682, 2523, 2523,  279,  279,
        279,  279, 2524, 2524, 2525, 2525,  466,  466, 1304, 2526,   76,
         76,   76,   76,   76,   76,   76,   76,   87,   87,   87,   87,
        863,  863,   76,   76,  582,  582, 2527, 2527,  230,  230,  230,
        230,  318,  318,  318,  318,   63,   63,   63,   63,   12,   12,
         12,   12,   12,   12,   12,   12,   52,   52,   52,   52, 1683,
       1683,  239,  239,  319,  319, 2528, 2528,   52,   52,   52,   52,
        134,  134,  134,  134,   12,   12,   12,   12,   17,   17,   17,
         17,   17,   17,   17,   17])
{'67625943': 1,
 '69666250': 2,
 '71676255': 3,
 '71686452': 4,
 '69646145': 5,
 '69656053': 6,
 '74665750': 7,
 '66625750': 8,
 '67646048': 9,
 '65605741': 10,
 '70676255': 11,
 '69646045': 12,
 '72655753': 13,
 '67645952': 14,
 '67625843': 15,
 '74675955': 16,
 '64595652': 17,
 '70656246': 18,
 '69625450': 19,
 '72645548': 20,
 '65625750': 21,
 '73696457': 22,
 '71645652': 23,
 '72676448': 24,
 '64615745': 25,
...
 '73645756': 997,
 '71625647': 998,
 "69615245'": 999,
 '64625747': 1000,
 ...}
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
4419
['72696551']
array([[949]])
  
  ```python:
# tokenize by "word" (ie chord)
tokenizer = keras.preprocessing.text.Tokenizer() 

tokenizer.fit_on_texts([str(train_copy)])
```

  
  
  ```python:
len(tokenizer.word_index)
```

  4419

  
  ```python:
for i in range(len(train_copy)):
  [train_copy[i]] = np.array(tokenizer.texts_to_sequences([train_copy[i]])) - 1
  # again if you don't convert to np.array and subtract one your NN goes to nan
```

  
  
  ```python:
len(train_copy)
train_copy[0]
```
Downloading data from https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz
122880/117661 [===============================] - 0s 0us/step
131072/117661 [=================================] - 0s 0us/step
'/root/.keras/datasets/bach/bach.tgz'
bach.tgz  test	train  valid
chorale_000.csv  chorale_058.csv  chorale_116.csv  chorale_174.csv
chorale_001.csv  chorale_059.csv  chorale_117.csv  chorale_175.csv
chorale_002.csv  chorale_060.csv  chorale_118.csv  chorale_176.csv
chorale_003.csv  chorale_061.csv  chorale_119.csv  chorale_177.csv
chorale_004.csv  chorale_062.csv  chorale_120.csv  chorale_178.csv
chorale_005.csv  chorale_063.csv  chorale_121.csv  chorale_179.csv
chorale_006.csv  chorale_064.csv  chorale_122.csv  chorale_180.csv
chorale_007.csv  chorale_065.csv  chorale_123.csv  chorale_181.csv
chorale_008.csv  chorale_066.csv  chorale_124.csv  chorale_182.csv
chorale_009.csv  chorale_067.csv  chorale_125.csv  chorale_183.csv
chorale_010.csv  chorale_068.csv  chorale_126.csv  chorale_184.csv
chorale_011.csv  chorale_069.csv  chorale_127.csv  chorale_185.csv
chorale_012.csv  chorale_070.csv  chorale_128.csv  chorale_186.csv
chorale_013.csv  chorale_071.csv  chorale_129.csv  chorale_187.csv
chorale_014.csv  chorale_072.csv  chorale_130.csv  chorale_188.csv
chorale_015.csv  chorale_073.csv  chorale_131.csv  chorale_189.csv
chorale_016.csv  chorale_074.csv  chorale_132.csv  chorale_190.csv
chorale_017.csv  chorale_075.csv  chorale_133.csv  chorale_191.csv
chorale_018.csv  chorale_076.csv  chorale_134.csv  chorale_192.csv
chorale_019.csv  chorale_077.csv  chorale_135.csv  chorale_193.csv
chorale_020.csv  chorale_078.csv  chorale_136.csv  chorale_194.csv
chorale_021.csv  chorale_079.csv  chorale_137.csv  chorale_195.csv
chorale_022.csv  chorale_080.csv  chorale_138.csv  chorale_196.csv
chorale_023.csv  chorale_081.csv  chorale_139.csv  chorale_197.csv
chorale_024.csv  chorale_082.csv  chorale_140.csv  chorale_198.csv
...
chorale_054.csv  chorale_112.csv  chorale_170.csv  chorale_228.csv
chorale_055.csv  chorale_113.csv  chorale_171.csv
chorale_056.csv  chorale_114.csv  chorale_172.csv
chorale_057.csv  chorale_115.csv  chorale_173.csv
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
[PosixPath('/root/.keras/datasets/bach/train/chorale_000.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_001.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_002.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_003.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_004.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_005.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_006.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_007.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_008.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_009.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_010.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_011.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_012.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_013.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_014.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_015.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_016.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_017.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_018.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_019.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_020.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_021.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_022.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_023.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_024.csv'),
...
 PosixPath('/root/.keras/datasets/bach/train/chorale_224.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_225.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_226.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_227.csv'),
 PosixPath('/root/.keras/datasets/bach/train/chorale_228.csv')]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:7: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  import sys
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:8: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:9: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  if __name__ == '__main__':
array([[74, 70, 65, 58],
       [74, 70, 65, 58],
       [74, 70, 65, 58],
       [74, 70, 65, 58],
       [75, 70, 58, 55],
       [75, 70, 58, 55],
       [75, 70, 60, 55],
       [75, 70, 60, 55],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 69, 62, 50],
       [77, 70, 62, 55],
       [77, 70, 62, 55],
       [77, 69, 62, 55],
       [77, 69, 62, 55],
       [75, 67, 63, 48],
       [75, 67, 63, 48],
       [75, 69, 63, 48],
       [75, 69, 63, 48],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [74, 70, 65, 46],
       [72, 69, 65, 53],
...
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46],
       [70, 65, 62, 46]])
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
(229,)
(192, 4)
(229,)
' 74706558 74706558 74706558 74706558 75705855 75705855 75706055 75706055 77696250 77696250 77696250 77696250 77706255 77706255 77696255 77696255 75676348 75676348 75696348 75696348 74706546 74706546 74706546 74706546 72696553 72696553 72696553 72696553 72696553 72696553 72696553 72696553 74706546 74706546 74706546 74706546 75696348 75696348 75676348 75676348 77656250 77656250 77656050 77656050 74675855 74675855 74675853 74675853 72675851 72675851 72675851 72675851 72655753 72655753 72655753 72655753 70656246 70656246 70656246 70656246 70656246 70656246 70656246 70656246 72696553 72696553 72696553 72696553 74715350 74715350 74715350 74715350 75725548 75725548 75725550 75725550 75676051 75676051 75676053 75676053 74676055 74676055 74675755 74675755 74655943 74655943 72635943 72635943 72635548 72635548 72635548 72635548 72635548 72635548 72635548 72635548 75676060 75676060 75676060 75676060 77706258 77706258 77706256 77706256 79706255 79706255 79706253 79706253 79706351 79706351 79706351 79706351 77706358 77706358 77706058 77706058 77706246 77706246 77686246 75686246 75675851 75675851 75675851 75675851 75675851 75675851 75675851 75675851 74675855 74675855 74675855 74675855 75675853 75675853 75675851 75675851 77655850 77655850 77655650 77655650 70635551 70635551 70635551 70635551 75656045 75656045 75656045 75656045 74655846 74655846 74655846 74655846 72655753 72655753 72655753 72655753 72655753 72655753 72655753 72655753 74655858 74655858 74655858 74655858 75675857 75675857 75675855 75675855 77656057 77656057 77656053 77656053 74655858 74655858 74655858 74655858 72675851 72675851 72675851 72675851 72655753 72655753 72655753 72655753 70656246 70656246 70656246 70656246 70656246 70656246 70656246 70656246'
4419
array([ 102,  102,  102,  102,  862,  862, 1303, 1303,  425,  425,  425,
        425,  579,  579,  580,  580,  475,  475,  476,  476,   44,   44,
         44,   44,   36,   36,   36,   36,   36,   36,   36,   36,   44,
         44,   44,   44,  476,  476,  475,  475,  338,  338, 1023, 1023,
         87,   87,  581,  581,  134,  134,  134,  134,   12,   12,   12,
         12,   17,   17,   17,   17,   17,   17,   17,   17,   36,   36,
         36,   36, 1679, 1679, 1679, 1679, 2521, 2521, 2522, 2522,  426,
        426, 1024, 1024,  393,  393, 1266, 1266,  465,  465, 1680, 1680,
        169,  169,  169,  169,  169,  169,  169,  169,  339,  339,  339,
        339,  340,  340, 1681, 1681, 1682, 1682, 2523, 2523,  279,  279,
        279,  279, 2524, 2524, 2525, 2525,  466,  466, 1304, 2526,   76,
         76,   76,   76,   76,   76,   76,   76,   87,   87,   87,   87,
        863,  863,   76,   76,  582,  582, 2527, 2527,  230,  230,  230,
        230,  318,  318,  318,  318,   63,   63,   63,   63,   12,   12,
         12,   12,   12,   12,   12,   12,   52,   52,   52,   52, 1683,
       1683,  239,  239,  319,  319, 2528, 2528,   52,   52,   52,   52,
        134,  134,  134,  134,   12,   12,   12,   12,   17,   17,   17,
         17,   17,   17,   17,   17])
{'67625943': 1,
 '69666250': 2,
 '71676255': 3,
 '71686452': 4,
 '69646145': 5,
 '69656053': 6,
 '74665750': 7,
 '66625750': 8,
 '67646048': 9,
 '65605741': 10,
 '70676255': 11,
 '69646045': 12,
 '72655753': 13,
 '67645952': 14,
 '67625843': 15,
 '74675955': 16,
 '64595652': 17,
 '70656246': 18,
 '69625450': 19,
 '72645548': 20,
 '65625750': 21,
 '73696457': 22,
 '71645652': 23,
 '72676448': 24,
 '64615745': 25,
...
 '73645756': 997,
 '71625647': 998,
 "69615245'": 999,
 '64625747': 1000,
 ...}
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
4419
['72696551']
array([[949]])
  
  
  ```python:
tokenizer.word_index
```

  {'67625943': 1,
 '69666250': 2,
 '71676255': 3,
 '71686452': 4,
 '69646145': 5,
 '69656053': 6,
 '74665750': 7,
 '66625750': 8,
 '67646048': 9,
 '65605741': 10,
 '70676255': 11,
 '69646045': 12,
 '72655753': 13,
 '67645952': 14,
 '67625843': 15,
 '74675955': 16,
 '64595652': 17,
 '70656246': 18,
 '69625450': 19,
 '72645548': 20,
 '65625750': 21,
 '73696457': 22,
 '71645652': 23,
 '72676448': 24,
 '64615745': 25,
...
 '73645756': 997,
 '71625647': 998,
 "69615245'": 999,
 '64625747': 1000,
 ...}
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...len(tokenizer.word_index)
  
  ```python:
len(tokenizer.word_index)
```

  
  4419

  ```python:
tokenized = np.array(tokenizer.texts_to_sequences(["72696551 "]))
```

  
  
  ```python:
tokenizer.sequences_to_texts(tokenized)
```
['72696551']

  
  
  ```python:
train_copy[0]
tokenized
```

array([[949]])


```python:
  [tokenized] = np.array(tokenizer.texts_to_sequences([str(train_copy)]))

```
```python:
tokenized
```
  array([], dtype=float64)

```python:
train_copy[0]
```
  array([ 102,  102,  102,  102,  862,  862, 1303, 1303,  425,  425,  425,
        425,  579,  579,  580,  580,  475,  475,  476,  476,   44,   44,
         44,   44,   36,   36,   36,   36,   36,   36,   36,   36,   44,
         44,   44,   44,  476,  476,  475,  475,  338,  338, 1023, 1023,
         87,   87,  581,  581,  134,  134,  134,  134,   12,   12,   12,
         12,   17,   17,   17,   17,   17,   17,   17,   17,   36,   36,
         36,   36, 1679, 1679, 1679, 1679, 2521, 2521, 2522, 2522,  426,
        426, 1024, 1024,  393,  393, 1266, 1266,  465,  465, 1680, 1680,
        169,  169,  169,  169,  169,  169,  169,  169,  339,  339,  339,
        339,  340,  340, 1681, 1681, 1682, 1682, 2523, 2523,  279,  279,
        279,  279, 2524, 2524, 2525, 2525,  466,  466, 1304, 2526,   76,
         76,   76,   76,   76,   76,   76,   76,   87,   87,   87,   87,
        863,  863,   76,   76,  582,  582, 2527, 2527,  230,  230,  230,
        230,  318,  318,  318,  318,   63,   63,   63,   63,   12,   12,
         12,   12,   12,   12,   12,   12,   52,   52,   52,   52, 1683,
       1683,  239,  239,  319,  319, 2528, 2528,   52,   52,   52,   52,
        134,  134,  134,  134,   12,   12,   12,   12,   17,   17,   17,
         17,   17,   17,   17,   17])
```python:
[train_copy[0]] = tokenizer.texts_to_sequences([str(train_copy[0])])
```
```python:
train_copy[0]
```
```python:
# This code may be useful; working with the ragged tensor is a little different

# Have to create this function so that you don't take windows across two different chorales
def make_windows(chorale): 
        # Make a Dataset of each chorale (so you can .window it)
        ds = tf.data.Dataset.from_tensor_slices(chorale)
        # .window each chorale Dataset -- making a Dataset of Window-Datasets (ie nested Datasets) 
        ds = ds.window(65, shift=1, drop_remainder=True)
        # flat_map the Window-Datasets so you return a Dataset of window-tensors
        return ds.flat_map(lambda window : window.batch(65))

ds_train = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(train_copy))
# Now we have a Dataset of tokenized-chorale-Tensors 
# (each element of Dataset is a tokenized chorale Tensor)

ds_train = ds_train.flat_map(make_windows) 
# make_windows creates a Dataset of window-tensors for each chorale,
#    which we then need to flat_map to have one final Dataset of window-tensors
#    from all chorales in the training set
```
```python:
for element in ds_train.take(5):
  print(element)
```
```python:
tf.Tensor(
[  93   93   93   93   70   70  477  477   22   22  341  341   35   35
   35   35    7    7 1684 1684  320  320  320  320 2529 2529  427  427
  370  370  362  362  178  178   22   22  362  362  362  362  362  362
  362  362  362  362  362  362   21   21  865  865  174  174 2530 2530
 2531 2531 2532 2532  199  199  199  199 1305], shape=(65,), dtype=int32)
tf.Tensor(
[  93   93   93   70   70  477  477   22   22  341  341   35   35   35
   35    7    7 1684 1684  320  320  320  320 2529 2529  427  427  370
  370  362  362  178  178   22   22  362  362  362  362  362  362  362
  362  362  362  362  362   21   21  865  865  174  174 2530 2530 2531
 2531 2532 2532  199  199  199  199 1305 1305], shape=(65,), dtype=int32)
tf.Tensor(
[  93   93   70   70  477  477   22   22  341  341   35   35   35   35
    7    7 1684 1684  320  320  320  320 2529 2529  427  427  370  370
  362  362  178  178   22   22  362  362  362  362  362  362  362  362
  362  362  362  362   21   21  865  865  174  174 2530 2530 2531 2531
 2532 2532  199  199  199  199 1305 1305 1306], shape=(65,), dtype=int32)
tf.Tensor(
[  93   70   70  477  477   22   22  341  341   35   35   35   35    7
    7 1684 1684  320  320  320  320 2529 2529  427  427  370  370  362
  362  178  178   22   22  362  362  362  362  362  362  362  362  362
  362  362  362   21   21  865  865  174  174 2530 2530 2531 2531 2532
 2532  199  199  199  199 1305 1305 1306 1306], shape=(65,), dtype=int32)
tf.Tensor(
...
 1684 1684  320  320  320  320 2529 2529  427  427  370  370  362  362
  178  178   22   22  362  362  362  362  362  362  362  362  362  362
  362  362   21   21  865  865  174  174 2530 2530 2531 2531 2532 2532
  199  199  199  199 1305 1305 1306 1306  866], shape=(65,), dtype=int32)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```
```python:
# We now have these in a form similar to our Char-RNN on Shakespeare, but each
#    token represents a musical chord instead of a text character
# Finish creating the X and Y and preprocess etc

ds_train = ds_train.shuffle(10000)  
ds_train = ds_train.batch(32)
ds_train = ds_train.map(lambda windows_batch: 
                                    (windows_batch[:, :-1], windows_batch[:, 1:]))
```
```python:
for element in ds_train.take(5):
  print(element)
```
```python:
(<tf.Tensor: shape=(32, 64), dtype=int32, numpy=
array([[  83,  132,  132, ...,  435,  435,  435],
       [  33,   33,    0, ...,  786,  755,  755],
       [  18,   18,   18, ..., 1039,   89,   89],
       ...,
       [ 235,  235,  492, ...,   17,   17,   17],
       [  15,   15,   15, ...,   15,   15,   15],
       [1080, 1080,  101, ...,  573,  573,  573]], dtype=int32)>, <tf.Tensor: shape=(32, 64), dtype=int32, numpy=
array([[ 132,  132,  132, ...,  435,  435,  435],
       [  33,    0,    0, ...,  755,  755,  106],
       [  18,   18,   18, ...,   89,   89, 1844],
       ...,
       [ 235,  492,  492, ...,   17,   17,   17],
       [  15,   15,   15, ...,   15,   15,   60],
       [1080,  101,  101, ...,  573,  573,  573]], dtype=int32)>)
(<tf.Tensor: shape=(32, 64), dtype=int32, numpy=
array([[  75,   75, 1060, ...,    5,    5,    5],
       [1119,    1,    1, ...,  115,  115,  229],
       [1336,  486,  486, ..., 2850, 2850,    6],
       ...,
       [  70,   70,   70, ...,   59,   59,   70],
       [   5,    5,    5, ..., 2817, 2818, 2818],
       [ 318,  482,  482, ..., 1749, 1749,  318]], dtype=int32)>, <tf.Tensor: shape=(32, 64), dtype=int32, numpy=
array([[  75, 1060, 1060, ...,    5,    5,  442],
       [   1,    1, 1273, ...,  115,  229,  229],
...
       ...,
       [  44,   44,   44, ...,  260,  260,   87],
       [ 774,  774,   92, ...,   19,  440,  440],
       [  20,  347,  109, ...,   27,  194,  194]], dtype=int32)>)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```
```python:
ds_train = ds_train.prefetch(1)
```
```python:
n_diff_chords = len(tokenizer.word_index)
n_diff_chords
# this is a lot of different chords; instead of using tf.one_hot to make each
#    chord a 4000+-long vector, I'm going to use embedding, which also has the advantage 
#    of mapping "relevant" chords next to each other in space
```
  4419

```python:
# make deep copy of valid 
import copy
valid_copy = copy.deepcopy(valid)
valid_copy.shape
```
  
  (76,)

```python:
# Convert each chorale to a string with 8-digit integer IDs for chords

for i in range(len(valid)): # iterate through each chorale in valid
  str_chorale = ""
  for j in range(len(valid[i])): # iterate through each chord in chorale i
    str_chord = ""
    for k in range(4):  # iterate through 4 notes (k) in chord j of chorale i
      str_note = str(valid[i][j][k])
      str_chord = str_chord + str_note
      #print(str_note, str_chord, valid[i][j])
    str_chorale = str_chorale + " " + str_chord
  valid_copy[i] = str_chorale
```
```python:
  valid_copy[0]

```
  ' 72676048 72676048 72676048 72676048 72676448 72676448 72676450 72676450 72676452 72676452 72676453 72676453 71676255 71676255 71656255 71656255 69646045 69646045 69646047 69646047 69646048 69646048 69656050 69656050 67676052 67676052 67676052 67676052 72646057 72646057 72646057 72646057 72725552 72725552 72725552 72725552 74716755 74716755 74716755 74716755 76676448 76676448 76676448 76676448 76676448 76676448 76676448 76676448 76676048 76676048 76676048 76676048 76676048 76676048 76676050 76676050 76676052 76676052 76676053 76676053 76675955 76675955 76675757 76675757 74655956 74655956 74655954 74655954 74645956 74645956 74645952 74645952 72646057 72646057 72646055 72646055 77696053 77696053 77696053 77696053 77676052 77676052 77655950 77655950 76726748 76726748 76726748 76726748 74716755 74716755 74716755 74716755 74716755 74716755 74716755 74716755 72646057 72646057 72666057 72666057 74675955 74675955 74675953 74675953 74675952 74675952 74675950 74675950 76676048 76676048 76676050 76676050 74685952 74685952 74685950 74685950 74685948 74685948 74685947 74685947 72696045 72696045 72696045 72696045 69696053 69696053 69696053 69696053 71675950 71675950 71656050 71656050 71676255 71676255 71676255 71676255 72676448 72676448 72676448 72676448 72676448 72676448 72676448 72676448 79676247 79676247 79676247 79676247 76676048 76676048 76676047 76676047 76675548 76675548 74655550 74655550 72676052 72676052 72676048 72676048 74696053 74696053 74696052 74696052 74675953 74675953 76675955 76675955 77725757 77725757 77725759 77725759 76725560 76725560 76725560 76725560 74716753 74716753 74696753 74696753 74676753 74676753 74676553 74676553 72676448 72676448 72676448 72676448 72676448 72676448 72676448 72676448 72676448 72676448 72676448 72676448'

```python:
  for i in range(len(valid_copy)):
  [valid_copy[i]] = np.array(tokenizer.texts_to_sequences([valid_copy[i]])) - 1
  # again if you don't convert to np.array and subtract one your NN goes to nan

```
  
  {'67625943': 1,
 '69666250': 2,
 '71676255': 3,
 '71686452': 4,
 '69646145': 5,
 '69656053': 6,
 '74665750': 7,
 '66625750': 8,
 '67646048': 9,
 '65605741': 10,
 '70676255': 11,
 '69646045': 12,
 '72655753': 13,
 '67645952': 14,
 '67625843': 15,
 '74675955': 16,
 '64595652': 17,
 '70656246': 18,
 '69625450': 19,
 '72645548': 20,
 '65625750': 21,
 '73696457': 22,
 '71645652': 23,
 '72676448': 24,
 '64615745': 25,
...
 '73645756': 997,
 '71625647': 998,
 "69615245'": 999,
 '64625747': 1000,
 ...}
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```python:
len(tokenizer.word_index)
```
  4419

```python:
tokenized = np.array(tokenizer.texts_to_sequences(["72696551"]))
```
['72696551']

```python:
valid_copy[0]
tokenized
```
  array([[949]])



```python:
[tokenized] = np.array(tokenizer.texts_to_sequences([str(valid_copy)]))
```


```python:
tokenized
```
array([], dtype=float64)


```python:
[valid_copy[0]] = tokenizer.texts_to_sequences([str(valid_copy[0])])
```


```python:
valid_copy[0]
```


```python:
# This code may be useful; working with the ragged tensor is a little different


ds_valid = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(valid_copy))
# Now we have a Dataset of tokenized-chorale-Tensors 
# (each element of Dataset is a tokenized chorale Tensor)

ds_valid = ds_valid.flat_map(make_windows) 
# make_windows creates a Dataset of window-tensors for each chorale,
#    which we then need to flat_map to have one final Dataset of window-tensors
#    from all chorales in the validation set
```


```python:
# We now have these in a form similar to our Char-RNN on Shakespeare, but each
#    token represents a musical chord instead of a text character
# Finish creating the X and Y and preprocess etc

ds_valid = ds_valid.shuffle(10000)  
ds_valid = ds_valid.batch(1000)
ds_valid = ds_valid.map(lambda windows_batch: 
                                    (windows_batch[:, :-1], windows_batch[:, 1:]))
```


```python:
for element in ds_valid.take(5):
  print(element)
```

(<tf.Tensor: shape=(1000, 64), dtype=int32, numpy=
array([[  48,   48,   48, ...,  544,   66,   66],
       [  39,   39,   64, ...,    1,    1,   33],
       [ 435,  435,  435, ..., 1667,  436, 1852],
       ...,
       [  12,   12,   12, ...,   52,   52,   52],
       [ 248,  719,  719, ...,   70,   70,   70],
       [1372,   23,   23, ..., 1212, 1212,   92]], dtype=int32)>, <tf.Tensor: shape=(1000, 64), dtype=int32, numpy=
array([[  48,   48,   48, ...,   66,   66,  273],
       [  39,   64,   64, ...,    1,   33,   33],
       [ 435,  435,  435, ...,  436, 1852, 1852],
       ...,
       [  12,   12,  115, ...,   52,   52,   52],
       [ 719,  719,  719, ...,   70,   70,   70],
       [  23,   23,  228, ..., 1212,   92,   92]], dtype=int32)>)
(<tf.Tensor: shape=(1000, 64), dtype=int32, numpy=
array([[  94,   94,   94, ...,  173,  173,  173],
       [  83,   83,   83, ...,  230,  230,   83],
       [ 101,  101, 1478, ...,  378,  378,  378],
       ...,
       [  48,   48,   48, ...,  854,  854,    2],
       [ 528,   89,   89, ..., 1347,  377,  377],
       [  99,   99,   99, ...,  743, 2499,   30]], dtype=int32)>, <tf.Tensor: shape=(1000, 64), dtype=int32, numpy=
array([[  94,   94,   94, ...,  173,  173,  173],
       [  83,   83,   83, ...,  230,   83,   83],
...
       ...,
       [1384, 1384, 1384, ...,   53,   53,   53],
       [2318,  231,  231, ..., 2318,  231,  231],
       [ 371,  371, 4139, ...,   90,  488,  488]], dtype=int32)>)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```python:
ds_valid = ds_valid.prefetch(1)
```


```python:
ds_valid
```
<PrefetchDataset element_spec=(TensorSpec(shape=(None, None), dtype=tf.int32, name=None), TensorSpec(shape=(None, None), dtype=tf.int32, name=None))>

```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])

```


```python:
  %%time
optimizer = keras.optimizers.Nadam(lr=1e-3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model.fit(ds_train, epochs=20, callbacks=[keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)])#, validationn_data=test_set)

```
Epoch 1/20
/usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/nadam.py:73: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(Nadam, self).__init__(name, **kwargs)
1264/1264 [==============================] - 98s 70ms/step - loss: 1.9460 - accuracy: 0.6921
Epoch 2/20
1264/1264 [==============================] - 89s 69ms/step - loss: 0.3132 - accuracy: 0.9452
Epoch 3/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.2337 - accuracy: 0.9595
Epoch 4/20
1264/1264 [==============================] - 89s 69ms/step - loss: 0.2112 - accuracy: 0.9633
Epoch 5/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1971 - accuracy: 0.9656
Epoch 6/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1878 - accuracy: 0.9674
Epoch 7/20
1264/1264 [==============================] - 87s 68ms/step - loss: 0.1819 - accuracy: 0.9683
Epoch 8/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1770 - accuracy: 0.9691
Epoch 9/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1727 - accuracy: 0.9699
Epoch 10/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1716 - accuracy: 0.9698
Epoch 11/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1678 - accuracy: 0.9705
Epoch 12/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1657 - accuracy: 0.9708
Epoch 13/20
1264/1264 [==============================] - 88s 69ms/step - loss: 0.1639 - accuracy: 0.9710
...
Epoch 18/20
1264/1264 [==============================] - 88s 68ms/step - loss: 0.1995 - accuracy: 0.9596
CPU times: user 23min 12s, sys: 2min 19s, total: 25min 32s
Wall time: 27min 29s
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

```python:
from google.colab import drive
drive.mount('/content/gdrive/')
```
Mounted at /content/gdrive/


```python:
model.save('/content/gdrive/MyDrive/Temp/Bach_RNN')
```
WARNING:absl:Found untraced functions such as gru_cell_layer_call_fn, gru_cell_layer_call_and_return_conditional_losses, gru_cell_1_layer_call_fn, gru_cell_1_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: /content/gdrive/MyDrive/Temp/Bach_RNN/assets
INFO:tensorflow:Assets written to: /content/gdrive/MyDrive/Temp/Bach_RNN/assets
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f319471f250> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f3194712690> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.

  
  
  
  
  

**Part 1**</br>
Process your validation set so that you can use it in training your RNN. Then utilize it, and EarlyStopping, and re-train the same RNN from Part 0.2.



```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])

```


```python:
%%time
optimizer = keras.optimizers.Nadam(lr=1e-3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model.fit(ds_valid, epochs=20, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)
```
Epoch 1/20
/usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/nadam.py:73: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(Nadam, self).__init__(name, **kwargs)
12/12 [==============================] - 23s 1s/step - loss: 7.2505 - accuracy: 0.0223 - val_loss: 6.6939 - val_accuracy: 0.0177
Epoch 2/20
12/12 [==============================] - 18s 1s/step - loss: 6.5992 - accuracy: 0.0193 - val_loss: 6.6476 - val_accuracy: 0.0119
Epoch 3/20
12/12 [==============================] - 18s 1s/step - loss: 6.5672 - accuracy: 0.0195 - val_loss: 6.6275 - val_accuracy: 0.0202
Epoch 4/20
12/12 [==============================] - 18s 1s/step - loss: 6.5581 - accuracy: 0.0189 - val_loss: 6.6381 - val_accuracy: 0.0144
Epoch 5/20
12/12 [==============================] - 18s 1s/step - loss: 6.5532 - accuracy: 0.0188 - val_loss: 6.6224 - val_accuracy: 0.0100
Epoch 6/20
12/12 [==============================] - 18s 1s/step - loss: 6.5422 - accuracy: 0.0181 - val_loss: 6.6065 - val_accuracy: 0.0107
Epoch 7/20
12/12 [==============================] - 18s 1s/step - loss: 6.5409 - accuracy: 0.0182 - val_loss: 6.6071 - val_accuracy: 0.0202
Epoch 8/20
12/12 [==============================] - 18s 1s/step - loss: 6.5382 - accuracy: 0.0186 - val_loss: 6.6295 - val_accuracy: 0.0110
Epoch 9/20
12/12 [==============================] - 18s 1s/step - loss: 6.5389 - accuracy: 0.0179 - val_loss: 6.6041 - val_accuracy: 0.0202
Epoch 10/20
12/12 [==============================] - 18s 1s/step - loss: 6.5311 - accuracy: 0.0189 - val_loss: 6.5914 - val_accuracy: 0.0143
Epoch 11/20
12/12 [==============================] - 18s 1s/step - loss: 6.5314 - accuracy: 0.0177 - val_loss: 6.6339 - val_accuracy: 0.0142
Epoch 12/20
12/12 [==============================] - 18s 1s/step - loss: 6.5264 - accuracy: 0.0183 - val_loss: 6.5619 - val_accuracy: 0.0202
Epoch 13/20
12/12 [==============================] - 18s 1s/step - loss: 6.5118 - accuracy: 0.0171 - val_loss: 6.6022 - val_accuracy: 0.0080
...
Epoch 20/20
12/12 [==============================] - 18s 1s/step - loss: 6.1466 - accuracy: 0.0305 - val_loss: 6.1859 - val_accuracy: 0.0238
CPU times: user 3min 58s, sys: 21.6 s, total: 4min 20s
Wall time: 6min 3s
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

```python:
from google.colab import drive
drive.mount('/content/gdrive/')
```
Drive already mounted at /content/gdrive/; to attempt to forcibly remount, call drive.mount("/content/gdrive/", force_remount=True).


```python:
model.save('/content/gdrive/MyDrive/Temp/Bach_RNN')
```

WARNING:absl:Found untraced functions such as gru_cell_2_layer_call_fn, gru_cell_2_layer_call_and_return_conditional_losses, gru_cell_3_layer_call_fn, gru_cell_3_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: /content/gdrive/MyDrive/Temp/Bach_RNN/assets
INFO:tensorflow:Assets written to: /content/gdrive/MyDrive/Temp/Bach_RNN/assets
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f31912e1dd0> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f3190ffc710> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
  
  
  **Part 2**</br>
Generate music (and not just chords), by doing what we did in class: feeding the predicted chord along with the previous chords back into the model to predict the next chord, etc. </br>
Create a couple new "chorales" utilizing the first chord from chorales in the test set to generate your music. 
  
```python:
from google.colab import drive
drive.mount('/content/gdrive/')
```

Drive already mounted at /content/gdrive/; to attempt to forcibly remount, call drive.mount("/content/gdrive/", force_remount=True).

```python:
import_model = keras.models.load_model('/content/gdrive/MyDrive/Temp/Bach_RNN')
```


```python:
seed_chord = test[0][4]
```


```python:
seed_chord
```


```python:
seed_chord
```


```python:
# Convert each chorale to a string with 8-digit integer IDs for chords
str_chord = ""
for i in range(4):  # iterate through 4 notes (k) in chord j of chorale i
  str_note = str(seed_chord[i])
  str_chord = str_chord + str_note
  #print(str_note, str_chord, test[i][j])
```


```python:
str_chord
```
'72605552'


```python:
tokenizer.word_index
```
{'67625943': 1,
 '69666250': 2,
 '71676255': 3,
 '71686452': 4,
 '69646145': 5,
 '69656053': 6,
 '74665750': 7,
 '66625750': 8,
 '67646048': 9,
 '65605741': 10,
 '70676255': 11,
 '69646045': 12,
 '72655753': 13,
 '67645952': 14,
 '67625843': 15,
 '74675955': 16,
 '64595652': 17,
 '70656246': 18,
 '69625450': 19,
 '72645548': 20,
 '65625750': 21,
 '73696457': 22,
 '71645652': 23,
 '72676448': 24,
 '64615745': 25,
...
 '73645756': 997,
 '71625647': 998,
 "69615245'": 999,
 '64625747': 1000,
 ...}
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...


```python:
np_chord = np.array([str_chord])
np_chord.shape
```
(1,)

```python:

X = np.array(tokenizer.texts_to_sequences(np_chord)) - 1
tf.one_hot(X, len(tokenizer.word_index))
```

<tf.Tensor: shape=(1, 1, 4419), dtype=float32, numpy=array([[[0., 0., 0., ..., 0., 0., 0.]]], dtype=float32)>

```python:
len(tokenizer.word_index)
```

4419

```python:
prediction = import_model.predict(X)
```


```python:
pred_class = np.argmax(prediction, axis=-1)
pred_class
```
array([[1]])


```python:
tokenizer.sequences_to_texts(pred_class)
```

['67625943']

  
```python:
75696048
```
75696048
```python:
def gen_chord1(chord):

  X = np.array(tokenizer.texts_to_sequences(chord)) - 1
  tf.one_hot(X, len(tokenizer.word_index)) 
  pred = np.argmax(import_model.predict(X), axis=-1)
  return tokenizer.sequences_to_texts(pred+2)[-1]

```

```python:
import random

def gen_chorale(chord, n):
  # Convert each chorale to a string with 8-digit integer IDs for chords
  str_chord = ""
  for i in range(4):  # iterate through 4 notes (k) in chord j of chorale i
    str_note = str(chord[i])
    str_chord = str_chord + str_note

  np_chord = np.array([str_chord])

  chorale = np_chord[0]
  for i in range(n):
    chord = gen_chord1(np_chord)
    np_chord = np.array([chord])
    chorale += " " + chord
  return chorale

choralesSample = []
for i in range(3):    
  choralesSample.append(gen_chorale(test[i][0], random.randint(100, 300)))      #use the starting chord from the first five chorales as seeds, and generate a random number for the different chorale length
  choralesSample[i] = choralesSample[i].replace("' ", "")
```

```python:
  choralesSample

```
  ['65605753 69625450 71676255 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048',
 '69666154 71676255 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048',
 '73686153 71676255 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048 74676249 67646048']

  
  
  
  
  **Part 3**</br>
Listen to your generated music. This will require converting the tokenized chords back into chords and feeding them into the player functions in Part 0.1.
<br/>
*Note: tokenizer is quite picky and you usually have to pass a list to `texts_to_sequences` or `sequences_to_texts`*
  
  
  
```python:
  # This code may be useful

def convert_tokens(input_music):
  
  mus = input_music.split(' ')
  music = np.empty((len(mus),4), dtype='int64')
  for i in range(len(mus)):
    mus[i] = int(mus[i])
    music[i,3] = mus[i] % 100
    mus[i] = (mus[i] - music[i,3]) / 100
    music[i,2] = mus[i] % 100
    mus[i] = (mus[i] - music[i,2]) / 100
    music[i,1] = mus[i] % 100
    mus[i] = (mus[i] - music[i,1]) / 100
    music[i,0] = mus[i] % 100
  return music

```
  

```python:
music = []
for i in range(len(choralesSample)):
  music.append(convert_tokens(choralesSample[i]))
  play_chords(music[i])
```

  **Part 4**</br>
You will notice that the music generator tends to "play it safe" and just repeats the same note over and over again. Therefore, create a temperature, like we did in class for the Char-RNN, that helps it pick the chord that is the second, third, etc choice according to their respective probabilities and the temperature (how "risky" we want the music generator to be)</br>
Again try it out on several chorales from the test set. Also try out a few first chords from the training set and see if it makes a difference (ie is our model overfit?).
  
  
  
```python:
#New chords by feeding str_test as a starter 

def gen_chord(chord, temperature):

  
  X = np.array(tokenizer.texts_to_sequences(chord)) - 1    
  tf.one_hot(X, len(tokenizer.word_index))  

  prediction = import_model.predict(X)       
  log_prob = tf.math.log(prediction[-1]) / temperature
  
  char_id = tf.random.categorical(log_prob, num_samples=1) 
  return tokenizer.sequences_to_texts(char_id.numpy()+1)[0]


def gen_chorale(chord, temp, n):    #chord is just the size 4 array of the 1st chor of the chorale

  # Convert each chorale to a string with 8-digit integer IDs for chords
  str_chord = ""    
  for i in range(4):  # iterate through 4 notes (k) in chord j of chorale i
    str_note = str(chord[i])  
    str_chord = str_chord + str_note

  np_chord = np.array([str_chord])    #make the string a size one numpy
  chorale = np_chord[0]


  for i in range(n):
    chord = gen_chord(np_chord, temp)     
    np_chord = np.array([chord])
    chorale += " " + chord
  return chorale

```

```python:
  choralesSample = []
for i in range(5):    
  choralesSample.append(gen_chorale(test[i][0], 1.2, random.randint(100, 300)))      #use the starting chord from the first five chorales as seeds, and generate a random number for the different chorale length
  choralesSample[i] = choralesSample[i].replace("' ", "")

```

```python:
choralesSample
```
  
  ['65605753 67625954 72686452 71675552 67645952 71676243 70676449 70625753 69646149 74676450 69636051 71696347 62625955 62575450 72676452 70656255 71686252 64615745 67635843 72695750 65575341 74645754 67625943 71696255 71686452 66606045 71655747 71676255 67665952 66645750 74696457 67645552 69665947 69666145 77656258 67625943 77656258 72666257 72696452 76676248 69645952 69626050 71646048 71685249 67645848 76676050 62575450 69625552 73665749 74675959 76676052 72646057 64645548 71676255 71676254 64575550 73696445 71595652 76696461 71666259 67646048 69665750 72676255 73706454 67625852 71625555 71686452 72696450 66625754 66625754 67656048 66625750 69645548 72696354 68595652 69625462 68665949 72686551 62625450 77625750 68665950 71645652 72696445 74645943 75636048 74656246 67625843 72676457 79696454 65625748 72676348 71666252 69656055 67626247 74655346 67646449 72696355 69666260 69656050 77656258 75676048 69656041 74686452 67625855 66635747 71676252 66626259 77695350 76716748 71665951 71676243 64645548 72645745 73645745 69645953 77676259 75675858 69646043 77696255 71665751 74696255 75635855 63605357',
 '69666154 71676452 70676260 71696259 64575249 69646052 64595650 69666250 71666255 74676245 73676457 78696252 70646154 67585551 67605551 64615257 69696650 71625450 64615545 69666250 64555248 67675952 69696650 71625956 71676247 64615757 69696650 69666250 75676051 74625858 75706246 66625752 67625943 71646455 69625953 66625750 65625753 74625547 69676250 74675955 70625855 70626553 76676152 77696250 60555248 71626755 67676258 76696450 73696445 72696254 69666250 74696552 69625754 71696452 69665947 71626755 74696050 74676450 65655850 75666057 66605745 66625750 71686250 64615345 71676055 66625750 71645957 67626043 72696457 75676051 74705351 70676245 67625550 69696254 74666250 72646057 73665854 70656243 72646057 68666247 69666057 67645750 64645548 64606057 72695750 74645754 71686352 72635551 7065466269656055 74706255 67676250 74665754 71645652 64575550 69665750 69665952 69666250 70675843 72666245 72675848 67656048 65625543 66625752 69696650 69605754 64606057 76675955 62585346 66605745 72696254 69646048 76676060 72645757 73696445 69656041 77696255 67625843 72686348 67645547 72696255 74665750 69676450 67625955 71645944 71666452 69605955 66645750 69646149 76646157 74625956 76676157 69605245 66636045 68625653 66605750 74625754 71676255 69696650 68645950 67675851 73665754 67646048 71676247 74645754 64625957 62625447 72676448 71676252 72676448 74656246 72676048 72646048 78696162',
 '73686153 74655350 75666359 66635945 69676250 64595652 75676051 74645548 67625952 69666050 72676450 67615552 69626050 67605548 72686348 72646057 71625350 72666245 73696450 66665950 77675851 73646455 67585551 67656247 69666250 69625752 75676051 62575450 76715955 72655750 69676448 64595652 69626053 69646252 72696254 71665951 72696450 72696457 75665947 62625741 74676452 69696354 71665955 76676448 64595652 67645952 69625447 74656050 69676041 69646249 62575243 67646250 76716155 74655955 67625953 64625757 72696450 67625955 71686452 74706546 75726853 66625754 62575450 79676247 74696553 74675948 70676250 71666359 71676248 67645952 69656250 65605750 74655547 74686559 67606460 68635648 68665950 72635844 69696550 71666452 69666250 77696252 69615545 67635844 64615545 76715955 71645650 64625945 69576054 74656050 68656160 76676048 66625750 69676252 71645952 64595543 71676443 73696457 71686450 60555236 67606052 71665751 0000 73696254 72666052 62575450 67646158 62625552 67625551 72645552 72676455 73645754 70675553 76676248 71686240 69646053 76676448 71625640 71625956 64595650 67625743 74695954 76676055 66625950 71676450 67645950 73645745 67606460 79706448 69605551 69645248 73646149 74656146 66635747 71676452 65625750 66665750 67625855 70676246 67646060 65645750 69676045 72696448 74656050 70656253 77726556 71625543 74625547 69605248 69696448 71675552 73645747 72696557 74716759 77676259 72676057 77696250 70625555 73706449 72655653 66605550 69666262 75636048 68646052 69636057 74695753 69666250 69656253 72666057 66576250 76675948 71635447 69696550 76676060 67655851 69646250 64645849 72656053 77655760 72646057 64625548 71686247 74645754 70656246 72696457 75636048 69605753 69656253 60555236 77696450 72696450 67645952 69646047 64605548 74625754 67625546 66715943 67625547 67636048 75655753 66626147 69646145 67595543 71625652 64615257 76696448 67595543 69606254 6759554374645955 64615345 74696553 67626258 64605848 76666150 70605450 76666146 67625850 75676057 69666450 71676454 67625855 73665757 69666249 76676047 71696651 72676052 67645952 77696053 66635945 72686551 67625843 73666159 72656052 76666146 66576250 67625753 70675551 69656052 69656052 77696053 69655353 62575450 72646045 69696650 74696650 69616455 69696250 67625943 72696254 64605552 67625046 70655843 76685950 74655846 72696253 69676448 76645652 64625952 63605345 64625952 71676247 69696650 69676250 67646152 72695451 71666255 73645757 64615257 67646052',
 '74706558 72656250 67626246 72696254 69615754 67665952 64605548 74666257 66595751 73696154 77676247 62575450 64595652 67646448 72676452 74665647 66626255 69645248 71676243 69646152 71696243 71645552 76676055 69666052 71676452 67646057 71696250 67645952 72696450 69665750 66615750 67645752 66625755 72676052 74686259 74675947 69696650 66625950 67645249 71685952 67646449 65645752 70625450 67615745 69646053 69646161 77696255 71686450 67665952 60555248 74696659 66635959 71676243 74625754 74676250 69605341 72635755 69676252 67645552 64606057 74675955 74666250 71596255 60555248 69615245 66646045 69576449 76676448 69646045 71656250 72706553 71665955 64625548 69696650 72675752 74645757 69676052 71716456 67696258 69646152 69666250 74655346 69696448 74666247 69576254 64595552 64615757 71676254 74655850 62595555 69676448 74675750 77676255 69576449 67625551 73686553 76696461 62625447 69696650 66625750 67645952 72676452 69615752 67676052 74656248 71675552 69656053 69666250 71676248 79716255 72675551 74716457 69676348 67655860 71676052 73676452 75665748 72695753 66625754 65575350 72646057 73645745 70676255 69676450 76715955 69665752 67645943 72645747 71696550 72625754 73665754 76696057 74645956 72645757 74656246 69666250 71645955 67646045 71625947 69656051 69605752 69576449 74715955 67626043 71625947 72646057 69666057 76645749 74655346 74696550 64625548 67635947 69645749 74655748 74675955 64625952 71666452 71646455 76696054 66635957 74635546 67646045 67625547 64615757 66625750 73665956 74706551 67646158 73665854 69646053 71676250 68646145 64625952 71676243 73696445 71626755 67625547 71676448 71666255 71625943 75676360 73645842 79676460 74715955 70656048 67626043 0000 72676348 72716550 69646050 74665949 76696149 69646145 64595550 72666257 64645552 74706757 72695750 70625753 77726557 70646043 72696557 71626747 72675952 64615557 67615552 74625754 71646055 69646145 71676247 71686259 69595743 72696254 74715955 71655747 70636048 66646045 72645755 64615745 74686452 69696254 71676255 66645947 72625754 72676452 72666247 67625955 69666250 74655750 73676452 67645750 74675955 69696650 69666048 72676448 66576254 76696155 72696053 74666050 67646060 67646448 62575450 75636048 72696455 74676259 64615649 72645745 69696650 72635753 72655750 64615755 72696445 74655850 69645248 68635947 71676247 67646045 71625555 66625750 73675757 69625462 77686253 71665950 76696245 69625752 67595543 69625754 74696053',
 '67636048 74706556 62625750 69646247 65636056 67646054 75676051 69656053 67646252 71676452 67625950 64615540 66665950 69605752 70655855 72645748 69646145 77696053 69675752 67625447 69666448 67645952 74665949 74696547 72655857 68645649 76675949 72676357 71696347 67625947 69666250 72676245 75676051 74656258 74625959 74665757 69696449 70656256 69656258 67676243 71645943 72646452 68625947 75706348 71646245 79706363 70655850 74675858 70655850 77676060 64595552 65606057 67625743 72696245 71656149 70626255 69675750 72696053 77706562 71665451 69626050 67675952 72696455 72696254 67625943 72696245 74696654 71635447 73645850 71646256 74665760 76675950 74696253 71645652 67625843 72696457 77656258 71696550 65605753 74705351 67655943 71696452 73655649 69666151 74716657 69656051 72676255 66646157 76666146 74716649 72645552 71696452 72645745 74696457 71686452 71676247 64555248 66615754 64595650 71656256 76686149 70656246 74706556 62605543 72695753 71645752 69645749 64595640 68645940 71626755 74665959 66616146 71625944 69645249 69625347 72665752 73696444 67676250 65615752 67585543 76676460 79676052 71646057 67595043 71655750 71665750 76696047 72695345 74696547 66615457 72676448 72696453 69686449 68645940 74676259 75675848 70676363 73675852 67646055 69645760 64615745 62575450 62625447 64595548 76676157 74686259 66625755 74696457 71696550 74655346 62575450 64625745 74625754 74696654 74645848 78716762 64615757 69656045 72625754 70676255 66635945 69665963 71676255 67625943 74696457 73696157 73676452 71625450 69676255 71676255 69666262 72676351 76706149 71625547 68635548 65646045 72676450 66665751 71665756 72696053 64595652 64625955 76646060 69666250 66635747 67676251 69646145 74696457 72655745 67645952 64645548 67645547 72645552 71635754 72696657 72636048 69645549 71676052 66645947 71655747 70666257 69656058 70626255 73645649 68645945 64625545 69696254 71696452 76676460 73695749 72666045 68656149 68645952 62575753 67646052 69625450 67646045']

```python:
music = []
for i in range(len(choralesSample)):
  music.append(convert_tokens(choralesSample[i]))
  play_chords(music[i])
```
**Part 5** </br>
Change some things on your RNN (or features), retrain and generate new music and see if you can't get something that sounds better.
```python:
n_diff_chords = len(tokenizer.word_index)
n_diff_chords
# this is a lot of different chords; instead of using tf.one_hot to make each
#    chord a 4000+-long vector, I'm going to use embedding, which also has the advantage 
#    of mapping "relevant" chords next to each other in space
```

  
  
  4419
  
  
  
  ```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),

    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])

```
```python:
%%time
optimizer = keras.optimizers.Nadam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model.fit(ds_train, epochs=20, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)
```
  Epoch 1/20
/usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/nadam.py:73: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(Nadam, self).__init__(name, **kwargs)
1264/1264 [==============================] - 98s 74ms/step - loss: 6.2737 - accuracy: 0.0545 - val_loss: 6.8461 - val_accuracy: 0.0748
Epoch 2/20
1264/1264 [==============================] - 94s 73ms/step - loss: 4.4051 - accuracy: 0.2273 - val_loss: 6.6309 - val_accuracy: 0.1212
Epoch 3/20
1264/1264 [==============================] - 94s 73ms/step - loss: 3.0575 - accuracy: 0.4401 - val_loss: 6.5623 - val_accuracy: 0.1519
Epoch 4/20
1264/1264 [==============================] - 94s 73ms/step - loss: 2.1273 - accuracy: 0.6093 - val_loss: 6.5179 - val_accuracy: 0.1700
Epoch 5/20
1264/1264 [==============================] - 94s 73ms/step - loss: 1.5166 - accuracy: 0.7254 - val_loss: 6.5102 - val_accuracy: 0.1861
Epoch 6/20
1264/1264 [==============================] - 94s 73ms/step - loss: 1.1366 - accuracy: 0.7970 - val_loss: 6.5524 - val_accuracy: 0.1941
Epoch 7/20
1264/1264 [==============================] - 94s 73ms/step - loss: 0.8799 - accuracy: 0.8456 - val_loss: 6.5914 - val_accuracy: 0.2007
Epoch 8/20
1264/1264 [==============================] - 94s 73ms/step - loss: 0.7197 - accuracy: 0.8740 - val_loss: 6.6430 - val_accuracy: 0.2042
Epoch 9/20
1264/1264 [==============================] - 94s 73ms/step - loss: 0.6058 - accuracy: 0.8945 - val_loss: 6.6824 - val_accuracy: 0.2074
Epoch 10/20
1264/1264 [==============================] - 94s 73ms/step - loss: 0.5257 - accuracy: 0.9083 - val_loss: 6.7007 - val_accuracy: 0.2147
CPU times: user 13min 20s, sys: 1min 21s, total: 14min 42s

```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model_2 = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),
    
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.Dropout(0.05),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])
```
```python:
%%time
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model_2.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model_2.fit(ds_train, epochs=20, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)
```
  
  
  Epoch 1/20
1264/1264 [==============================] - 166s 125ms/step - loss: 6.8093 - accuracy: 0.0211 - val_loss: 6.5483 - val_accuracy: 0.0457
Epoch 2/20
1264/1264 [==============================] - 158s 124ms/step - loss: 3.7480 - accuracy: 0.2843 - val_loss: 6.1072 - val_accuracy: 0.1321
Epoch 3/20
1264/1264 [==============================] - 158s 124ms/step - loss: 1.6354 - accuracy: 0.6316 - val_loss: 6.4640 - val_accuracy: 0.1441
Epoch 4/20
1264/1264 [==============================] - 158s 124ms/step - loss: 0.9395 - accuracy: 0.7844 - val_loss: 6.8191 - val_accuracy: 0.1496
Epoch 5/20
1264/1264 [==============================] - 158s 124ms/step - loss: 0.6829 - accuracy: 0.8445 - val_loss: 7.0888 - val_accuracy: 0.1531
Epoch 6/20
1264/1264 [==============================] - 157s 123ms/step - loss: 0.5409 - accuracy: 0.8769 - val_loss: 7.3209 - val_accuracy: 0.1640
Epoch 7/20
1264/1264 [==============================] - 156s 122ms/step - loss: 0.4595 - accuracy: 0.8945 - val_loss: 7.5413 - val_accuracy: 0.1554
CPU times: user 15min 24s, sys: 1min 15s, total: 16min 40s
Wall time: 18min 29s

```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model_3 = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),
    keras.layers.LayerNormalization(),                #supposed to work better than batch normalization for RNNs
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])
```
```python:
  %%time
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model_3.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model_3.fit(ds_train, epochs=20, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)

```
  Epoch 1/20
1264/1264 [==============================] - 165s 123ms/step - loss: 0.9258 - accuracy: 0.8371 - val_loss: 4.5239 - val_accuracy: 0.3817
Epoch 2/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1975 - accuracy: 0.9557 - val_loss: 4.6827 - val_accuracy: 0.3845
Epoch 3/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1399 - accuracy: 0.9682 - val_loss: 4.9571 - val_accuracy: 0.3763
Epoch 4/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1415 - accuracy: 0.9676 - val_loss: 5.1202 - val_accuracy: 0.3729
Epoch 5/20
1264/1264 [==============================] - 156s 122ms/step - loss: 0.1317 - accuracy: 0.9699 - val_loss: 5.3006 - val_accuracy: 0.3685
Epoch 6/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1309 - accuracy: 0.9698 - val_loss: 5.4245 - val_accuracy: 0.3649
CPU times: user 13min 3s, sys: 1min 2s, total: 14min 5s
Wall time: 15min 46s
```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model_4 = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),
    keras.layers.LayerNormalization(),                #supposed to work better than batch normalization for RNNs
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.Dropout(0.05),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])
```
```python:
  %%time
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model_4.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model_4.fit(ds_train, epochs=20, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)

```

Epoch 1/20
1264/1264 [==============================] - 165s 124ms/step - loss: 0.9160 - accuracy: 0.8385 - val_loss: 4.5830 - val_accuracy: 0.3725
Epoch 2/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1711 - accuracy: 0.9616 - val_loss: 4.7201 - val_accuracy: 0.3800
Epoch 3/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1414 - accuracy: 0.9682 - val_loss: 4.8879 - val_accuracy: 0.3773
Epoch 4/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1338 - accuracy: 0.9696 - val_loss: 5.1248 - val_accuracy: 0.3730
Epoch 5/20
1264/1264 [==============================] - 156s 123ms/step - loss: 0.1327 - accuracy: 0.9697 - val_loss: 5.2851 - val_accuracy: 0.3649
Epoch 6/20
1264/1264 [==============================] - 157s 123ms/step - loss: 0.1285 - accuracy: 0.9704 - val_loss: 5.4470 - val_accuracy: 0.3638
CPU times: user 13min 5s, sys: 1min 4s, total: 14min 10s
Wall time: 15min 47s

```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model_5 = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),
    keras.layers.LayerNormalization(),                #supposed to work better than batch normalization for RNNs
    keras.layers.LSTM(1028, return_sequences=True),
    keras.layers.LSTM(1028, return_sequences=True),
    keras.layers.LSTM(1028, return_sequences=True),
    keras.layers.Dropout(0.05),
    keras.layers.LSTM(1028, return_sequences=True),
    keras.layers.LSTM(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])%%time
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model_5.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model_5.fit(ds_train, epochs=5, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)
```
```python:
%%time
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model_5.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model_5.fit(ds_train, epochs=5, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)
```

Epoch 1/5
1264/1264 [==============================] - 238s 178ms/step - loss: 0.6689 - accuracy: 0.8597 - val_loss: 7.8242 - val_accuracy: 0.1179
Epoch 2/5
1264/1264 [==============================] - 224s 176ms/step - loss: 0.4351 - accuracy: 0.9056 - val_loss: 8.0740 - val_accuracy: 0.1256
Epoch 3/5
1264/1264 [==============================] - 224s 176ms/step - loss: 0.3073 - accuracy: 0.9307 - val_loss: 8.2731 - val_accuracy: 0.1361
Epoch 4/5
1264/1264 [==============================] - 224s 176ms/step - loss: 0.2796 - accuracy: 0.9354 - val_loss: 8.3683 - val_accuracy: 0.1392
Epoch 5/5
1264/1264 [==============================] - 225s 177ms/step - loss: 0.2152 - accuracy: 0.9488 - val_loss: 8.4868 - val_accuracy: 0.1425
CPU times: user 15min 36s, sys: 1min 1s, total: 16min 37s
Wall time: 18min 53s

```python:
# Use Embeddings with output_dim = 8 --> rule of thumb is fourth root of input_dim
model_6 = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_diff_chords, output_dim=8,
                           input_shape=[None]),
    keras.layers.LayerNormalization(),                #supposed to work better than batch normalization for RNNs
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.LayerNormalization(),               

    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.GRU(1028, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_diff_chords, 
                                                    activation="softmax"))
])
```
  
  
```python:
%%time
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model_6.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model_6.fit(ds_train, epochs=5, validation_data=ds_valid,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]) #, validation_data=valid_set)
```
  
  Epoch 1/5
1264/1264 [==============================] - 200s 150ms/step - loss: 0.8333 - accuracy: 0.8538 - val_loss: 4.3950 - val_accuracy: 0.3859
Epoch 2/5
1264/1264 [==============================] - 189s 148ms/step - loss: 0.1499 - accuracy: 0.9679 - val_loss: 4.6508 - val_accuracy: 0.3729
Epoch 3/5
1264/1264 [==============================] - 189s 148ms/step - loss: 0.1412 - accuracy: 0.9688 - val_loss: 4.8727 - val_accuracy: 0.3716
Epoch 4/5
1264/1264 [==============================] - 189s 149ms/step - loss: 0.1363 - accuracy: 0.9695 - val_loss: 5.0624 - val_accuracy: 0.3674
Epoch 5/5
1264/1264 [==============================] - 189s 148ms/step - loss: 0.1341 - accuracy: 0.9696 - val_loss: 5.2128 - val_accuracy: 0.3671
CPU times: user 13min 14s, sys: 1min 1s, total: 14min 15s
Wall time: 15min 55s
```python:
  
  from google.colab import drive
drive.mount('/content/gdrive/')

```
  
  Drive already mounted at /content/gdrive/; to attempt to forcibly remount, call drive.mount("/content/gdrive/", force_remount=True).

```python:
model_6.save('/content/gdrive/MyDrive/Temp/Bach_RNN_experiment')
```
  WARNING:absl:Found untraced functions such as gru_cell_20_layer_call_fn, gru_cell_20_layer_call_and_return_conditional_losses, gru_cell_21_layer_call_fn, gru_cell_21_layer_call_and_return_conditional_losses, gru_cell_22_layer_call_fn while saving (showing 5 of 10). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: /content/gdrive/MyDrive/Temp/Bach_RNN_experiment/assets
INFO:tensorflow:Assets written to: /content/gdrive/MyDrive/Temp/Bach_RNN_experiment/assets
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f2cfbdc7c10> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f311649afd0> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f2cf2425210> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f2cf23f09d0> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<keras.layers.recurrent.GRUCell object at 0x7f2cfbdc6ad0> has the same name 'GRUCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.GRUCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
  
  
  
  Model Testing

  
  
  
```python:
  import_model = keras.models.load_model('/content/gdrive/MyDrive/Temp/Bach_RNN_experiment')

```
  
  //Retrieved rom RNN file on moodle
  
  
```python:
  
  import random

choralesSample = []
for i in range(5):    
  choralesSample.append(gen_chorale(test[i][0], 2, random.randint(100, 300)))      
  choralesSample[i] = choralesSample[i].replace("' ", "")

```
```python:
music = []
for i in range(len(choralesSample)):
  music.append(convert_tokens(choralesSample[i]))
  play_chords(music[i])
```
  
  All the models did great, However the last model (model_6) was the best one with the accuracy of 96.96%. As the RAM was filling out quick, so i reduced the number of epoch for last two models to 5.
