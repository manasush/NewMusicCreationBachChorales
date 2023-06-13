# New Music Creation From Bach Chorales
Using the dataset containing 382 chorales by Johann Sebastian Bach and generating new music




Dataset contains 382 chorales by Johann Sebastian Bach (in the public domain), where each chorale is composed of 100 to 640 chords with a temporal resolution of 1/16th (ie each chorale is 100 to 640 sixteenth-notes long). Each chord has four notes, composed of 4 integers, each indicating the index of a note on a piano, except for the value 0 which means "no note played."

To get longer notes, chords are repeated (ie, if a chord is repeated twice it is equivalent to an 1/8 note, repeated four times it is equivalent to a quarter note, etc). Note that this doesn't truly represent music as in music you can repeat a note and it means to replay it, not to lengthen its duration. But, this is where we start for this project.
