# PixelProof API
-----
This is the API responsible for the detection of if an image is AI generated or not, it uses flask and tensorflow with a pretrained model that you will have to provide yourself (for now until i train my own later).

NOTE: I used the weights from (this)[https://github.com/MrBinit/ai-generated-image-detect-EfficientNetB4] repo for the detection mostly because i couldnt find any better options that were free (lots of paid APIs though if you're willing to fork over 20 bucks a month minimum), i plan on training my own CNN for this use case but for right now please make do with what you can find/make yourself.

**IMPORTANT**: this is not meant for production environments at all, there are no safety features, no rate limiting or authentication, deploying this in its current state will most likely get you DOSed by bots spamming it.
## Contributing
----
Please do, this was written partly by chatgpt bc I hate flask with all my being and i did not wanna touch it, the code can definitely be better.

