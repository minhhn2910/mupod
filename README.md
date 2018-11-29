Steps to setup and optimized bitwidth for your own CNN:

From the root folder of Mupod, do the following:

Prepare data  : check model prototxt file, and test data (imagenet), if you can run caffe test with imagenet then the following steps will run smoothly.
Steps to download and prepare validation set of imagenet is given in Caffe tutorial, also download alexnet pretrained weight: bvlc_reference_caffenet.caffemodel.
In setting up caffe and models, no modification needed to run Mupod. Note that all the below command, we use iteration = 1, all the iterations over batches are included in the code (half of the validation dataset), no need to specify --iteration here.

##### Step 1: Binary search for sigma_Y_L:

`./build/tools/caffe search -model=./examples/alexnet/alexnet_analyze.txt -weights=./examples/alexnet/bvlc_reference_caffenet.caffemodel --iterations 1 --gpu 0`

##### Step 2: Analyze lambda and theta for each layer: 
Modify caffe/net.cpp line-672: `float current_error = 1.0;` to the sigma_Y_L found in step 1 and do make all.
This is to stablize the lambda and theta values, different `current_error` may gives different lambda theta values (although they give similar or very slightly different bitwidth in the end). This is the guessing Delta_X_K in the paper that we did not have space to explain in details. The source file already has all sigma_Y_L for 1% loss of accuracy, can jump to this step without doing step 1 on already-analyzed CNNs.

`./build/tools/caffe analyze -model=./examples/alexnet/alexnet_analyze.txt -weights=./examples/alexnet/bvlc_reference_caffenet.caffemodel --iterations 1 --gpu 0`

##### Step 3: Get max absolute value of each layer => integer bitwidth :

`./build/tools/caffe analyze_integer -model=./examples/alexnet/alexnet_analyze.txt -weights=./examples/alexnet/bvlc_reference_caffenet.caffemodel --iterations 1 --gpu 0`

##### Step 4: Run the optimization script (need to modify the weightage of each layer in the script, and paste the analyzed lambda and theta in step 1 to this script. Some examples are given in the script):
`octave debugging/optimize_script.txt`

##### Step 5: Take the fractional bitwidth for each layer, add that fractional bitwidth with integer bitwidth for each layer in step 3 and deploy on your variable-biwidth hardware.

Test output accuracy (to make sure the bitwidths are correct):
modify file : debugging/bitwidth_solution.txt
the first number will be the Fractional bitwidth of weights. The following N number will be the fractional bitwidth of all N layers:
e.g. CNN 3 layer with fraction bitwidth = -2,0,1 ; weight bitwidth = 10 ; (No need to include integer bitwidth here), the content of bitwidth_solution should be :
10,-2,0,1

then, run the test for that bitwidth configuration to make sure accuracy is satisfied:
`./build/tools/caffe test_fixedpoint -model=./examples/alexnet/alexnet_analyze.txt -weights=./examples/alexnet/bvlc_reference_caffenet.caffemodel --iterations 1 --gpu 0`

##### Step 0: to get all the precomputed and optimized bitwidths for 1% and 5% error thresholds for all 8 CNNs

Simply run `python showbitwidth.py`

Author of this modification & method: Minh Ho. Please open pull request if anything goes wrong.

License, copyright and related documents follow the original CAFFE_README.md and caffe tool. 

TODO: 
parsing commandline argument so we don't need to recompile each time when we change accuracy.
fix compiler warnings.
