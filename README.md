# OtsuPyre
**Blazingly Fast Approximation of Otsu's Multi-threshold Method**

## Background

Otsu's Method is an algorithm for thresholding a grayscale image into a well-balanced black and white image by analyzing an image's histogram.

![input image](https://github.com/lancekindle/OtsuPyre/wiki/images/tractor.png  "input greyscale image")![output](https://github.com/lancekindle/OtsuPyre/wiki/images/tractor_bw.png  "BW output image")

It can also be extended to multithresholding, where multiple thresholds can be used to drastically reduce the number of grey shades in an image. Below is an example tractor rendered in only four shades of grey, as calculated by OtsuPyre.

![3 thresholds, 4 colors](https://github.com/lancekindle/OtsuPyre/wiki/images/tractor_four_tone.png  "3 thresholds, 4 colors")

## The Pyramid Method

The biggest slowdown with Otsu's Method is that its algorithm must iterate over the entire histogram for each threshold, making its complexity O(L<sup>M</sup>) where M is the number of thresholds, and L is the range of the intensitly levels (typically 256 for a standard image).

OtsuPyre takes a divide-and-conquer approach to multithresholding by recognizing that manipulating the histogram size will reduce iteration count and achieve higher speeds. OtsuPyre iteratively reduces the histogram to half its size until the histogram length, N, is minimally greater than or equal to M. Then OtsuPyre computes the thresholds on this minimal histogram. This first histograms length is bounded by the number of thresholds: M ≤ N ≤ 2M, and the complexity for this first computation is O(N<sup>L</sup>).

From there, OtsuPyre will:

1. Scale up the computed threshold by a factor of 2
2. Use a histogram twice as large as the previous one
3. Search larger histogram within error bounds of our previously computed thresholds (error bounds are calculated from the scaling factor in step 1 & 2)
4. Return new thresholds
5. Repeat steps 1 - 4 until thresholds have been calculated on original histogram.

Step 3 is integral to the speed of OtsuPyre. Assuming we are dealing with a standard image, the histogram and thresholds will both be scaled by 2, and approximate error bounds = 2. Therefore a new threshold search will search a 5x5 area around each threshold, making the complexity for each iterative calculation O(5<sup>M</sup>).

There are K iterations, which correlate to the original size of the histogram and the number of reductions / compressions taken before it was just small enough to fit the desired thresholds, which leaves the general complexity as O(N<sup>M</sup> + (8 - K) * 5<sup>M</sup>)

## Efficiency Numbers

Searching for M thresholds on a typical image with OtsuPyre will have a maximum complexity of ~ O((2M)<sup>M</sup> + 8 * 5<sup>M</sup>), and will require Z iterations. Here are some calculated numbers

- M(number of thresholds): Y(complexity) == Z(iterations)
- 2: O(2<sup>2</sup> + 7 * 5<sup>2</sup>) == 179
- 3: O(4<sup>3</sup> + 6 * 5<sup>3</sup>) == 814
- 4: O(4<sup>4</sup> + 6 * 5<sup>4</sup>) == 4,006
- 5: O(8<sup>5</sup> + 5 * 5<sup>5</sup>) == 48,393
- 6: O(8<sup>6</sup> + 5 * 5<sup>6</sup>) == 340,269
- 7: O(8<sup>7</sup> + 5 * 5<sup>7</sup>) == 2,487,777
- 8: O(8<sup>8</sup> + 5 * 5<sup>8</sup>) == 18,730,341
- 9: O(16<sup>9</sup> + 4 * 5<sup>9</sup>) == 68,727,289,236

Now compare to a naive Otsu implementation, which is O(256<sup>M</sup>)

- 1: 256<sup>1</sup> == 256
- 2: 256<sup>2</sup> == 65,536
- 3: 256<sup>3</sup> == 16,777,216
- 4: 256<sup>4</sup> == 4,294,967,296
- 5: 256<sup>5</sup> == 1,099,511,627,776

Which can be interpreted to say that Naive Otsu's Method can easily find 3 thresholds, while OtsuPyre can find 8. After those points, both algorithms quickly succumb to the exponential increase in computation time. 

Please note that OtsuPyre is an approximation algorithm. Although thresholds found give good separation between grey levels, they do not match thresholds found by a Naive Otsu's implementation.
