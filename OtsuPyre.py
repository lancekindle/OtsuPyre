import math
import cv2
import numpy as np


class OtsuPyramid(object):
    """ class segments histogram into pyramid of histograms, each half the size of the previous. Generating omega
    and mu values allow for this to work extremely fast
    """

    def load_image(self, im):
        self.im = im
        h, w = im.shape[:2]
        N = float(h * w)  # N = number of pixels in image
                # cast N as float, because we need float answers when dividing by N
        bins = 256  # bins = number of intensity levels
        images = [im]
        channels = [0]
        mask = None
        histBins = [bins]
        ranges = [0, bins]  # range of pixel values. I've tried setting this to im.min() and im.max() but I get errors...
        hist = cv2.calcHist(images, channels, mask, histBins, ranges)  # hist is a numpy array of arrays. So accessing hist[0]
                # gives us an array, which messes with calculating omega. So we convert np array to list of ints
        hist = [int(h) for h in hist]
        histPyr, omegaPyr, muPyr = self._create_histogram_and_stats_pyramids(hist)
        self.omegaPyramid = [omegas for omegas in reversed(omegaPyr)]  # so that pyramid[0] is the smallest pyramid
        self.muPyramid = [mus for mus in reversed(muPyr)]
        
    def _create_histogram_and_stats_pyramids(self, hist):
        """ expects hist to be a single list of numbers (no numpy array)
        """
        bins = len(hist)
        reductions = int(math.log(bins, 2))
        histPyramid = []
        omegaPyramid = []
        muPyramid = []
        for i in range(reductions):
            histPyramid.append(hist)
            reducedHist = [hist[i] + hist[i + 1] for i in range(0, bins, 2)]
            hist = reducedHist  # collapse a list to half its size, combining the two collpased numbers into one
            bins = bins / 2  # update bins to reflect the length of the new histogram
        for hist in histPyramid:
            omegas, mus, muT = self._calculate_omegas_and_mus_from_histogram(hist)
            omegaPyramid.append(omegas)
            muPyramid.append(mus)
        return histPyramid, omegaPyramid, muPyramid

    def _calculate_omegas_and_mus_from_histogram(self, hist):
        probabilityLevels, meanLevels = self._calculate_histogram_pixel_stats(hist)
        bins = len(probabilityLevels)
        ptotal = 0.0
        omegas = []  # sum of probability levels up to k
        for i in range(bins):
            ptotal += probabilityLevels[i]
            omegas.append(ptotal)
        mtotal = 0.0
        mus = []
        for i in range(bins):
            mtotal += meanLevels[i]
            mus.append(mtotal)
        muT = float(mtotal)  # muT is the total mean levels.
        return omegas, mus, muT

    def _calculate_histogram_pixel_stats(self, hist):
        bins = len(hist)  # bins = number of intensity levels
        N = float(sum(hist))  # N = number of pixels in image
        probabilityLevels = [hist[i] / N for i in range(bins)]  # percentage of pixels at each intensity level i
                # => P_i
        meanLevels = [i * probabilityLevels[i] for i in range(bins)]  # mean level of pixels at intensity level i
        return probabilityLevels, meanLevels                            # => i * P_i


class OtsuFastMultithreshold(OtsuPyramid):

    def calculate_k_thresholds(self, k):
        self.threshPyramid = []
        start = self._get_starting_pyramid_index(k)
        self.bins = len(self.omegaPyramid[start])
        thresholds = [self.bins / 2 for i in range(k)]  # first-guess thresholds will be half the size of bins
        deviate = self.bins / 2  # give hunting algorithm full range so that initial thresholds can become any value (0-bins)
        for i in range(start, len(self.omegaPyramid)):
            omegas = self.omegaPyramid[i]
            mus = self.muPyramid[i]
            hunter = ThresholdHunter(omegas, mus, deviate)
            thresholds = hunter.find_best_thresholds_around_original(thresholds)
            self.threshPyramid.append(thresholds)
            # now we scale-up thresholds and set deviate for hunting in limited area. Logic / experiments suggest that you
            # only need to deviate by up to 2 when scaling up the previous thresholds by 2.
            deviate = 2
            thresholds = [t * 2 for t in thresholds]  # list(np.array(thresholds) * 2)
        return [t / 2 for t in thresholds]  # return readjusted threshold (since it was scaled up by 2 incorrectly in last loop)
    
    def _get_starting_pyramid_index(self, k):
        """ given the number of thresholds, return the minium starting index
        of pyramid to use in calculating thresholds
        """
        return int(math.ceil( math.log(k + 1, 2) ))

    def apply_thresholds_to_image(self, thresholds, im=None):
        if im is None:
            im = self.im
        k = len(thresholds)
        thresholds = [None] + thresholds + [None]
        greyValues = [0] + [int(256 / k * (i + 1)) for i in range(0, k - 1)] + [255]  # I think you need to use 255 / k *...
        finalImage = np.zeros(im.shape, dtype=np.uint8)
        for i in range(k + 1):
            kSmall = thresholds[i]
            if kSmall:
                bw = (im >= kSmall)  # create a black-and-white "image" representing pixels between the two thresholds
            else:
                bw = np.ones(im.shape, dtype=np.bool8)
            kLarge = thresholds[i + 1]
            if kLarge:
                bw &= (im < kLarge)
            greyLevel = greyValues[i]
            greyImage = bw * greyLevel  # apply grey-coloring to black-and-white image
            finalImage += greyImage  # add grey portions to image. There should be no overlap between each greyImage added
        return finalImage
        

class ThresholdHunter(object):
    """ hunt around given thresholds in a small space to look for a better threshold
    """

    def __init__(self, omegas, mus, deviate=2):
        self.sigmaB = BetweenClassVariance(omegas, mus)
        self.bins = self.sigmaB.bins  # used to be called L
        self.deviate = deviate  # hunt 2 (or other amount) to either side of thresholds

    def find_best_thresholds_around_original(self, originalThresholds):
        """ this is the one you were last typing up. Given guesses for best threshold, explore to either side of the threshold
        and return the best result.
        """
        bestResults = (0, originalThresholds, [0 for t in originalThresholds])
        for thresholds, offsets in self._jitter_thresholds_and_offsets_generator(originalThresholds, 0, self.bins -  1):
            offsets =  [val + self.deviate for val in offsets]  # set offset range 0->5  (0 -> 2*offset + 1)
            variance = self.sigmaB.get_total_variance(thresholds)
            newResult = (variance, thresholds, offsets)
            bestResults = max(bestResults, newResult)
        return bestResults[1]

    def _jitter_thresholds_and_offsets_generator(self, thresholds, min_, max_):
        pastThresh = thresholds[0]
        if len(thresholds) == 1:
            for offset in range(-self.deviate, self.deviate + 1):  # -2 through +2
                thresh = pastThresh + offset
                if thresh < min_ or thresh > max_:
                    continue  # skip since we are conflicting with bounds
                yield [thresh], [offset]
        else:
            thresholds = thresholds[1:]  # new threshold without our threshold included
            m = len(thresholds)  # number of threshold left to generate in chain
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                if thresh < min_ or thresh + m > max_:  # verify we don't use the same value as the previous threshold
                    continue                         # and also verify our current threshold will not push the last threshold past max
                recursiveGenerator = self._jitter_thresholds_and_offsets_generator(thresholds, thresh + 1, max_)
                for otherThresholds, otherOffsets in recursiveGenerator:
                    yield [thresh] + otherThresholds, [offset] + otherOffsets


class BetweenClassVariance(object):

    def __init__(self, omegas, mus):
        self.omegas = omegas
        self.mus = mus
        self.bins = len(mus)  # number of bins / luminosity choices
        self.muTotal = sum(mus)

    def get_total_variance(self, thresholds):
        """ function will pad the thresholds argument with minimum and maximum thresholds to calculate
        between class variance
        """
        thresholds = [0] + thresholds + [self.bins - 1]
        numClasses = len(thresholds) - 1
        sigma = 0
        for i in range(numClasses):
            k1 = thresholds[i]
            k2 = thresholds[i+1]
            sigma += self._between_thresholds_variance(k1, k2)
        return sigma

    def _between_thresholds_variance(self, k1, k2):  # to be used in calculating between class variances only!
        omega = self.omegas[k2] - self.omegas[k1]
        mu = self.mus[k2] - self.mus[k1]
        muT = self.muTotal
        return omega * ( (mu - muT)**2)

if __name__ == '__main__':
    filename = 'boat.jpg'
    dot = filename.index('.')
    prefix, extension = filename[:dot], filename[dot:]
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    otsu = OtsuFastMultithreshold()
    otsu.load_image(im)
##    kThresholds = [22, 31, 39, 47, 58, 75, 87, 107]
    k = 1
    kThresholds = otsu.calculate_k_thresholds(k)
    print(kThresholds)
    crushed = otsu.apply_thresholds_to_image(kThresholds)
    cv2.imwrite(prefix + '_crushed_' + str(len(kThresholds)) + extension, crushed)
    # 5 = [54, 78, 104, 150, 190]
    # 6 = [48, 70, 90, 118, 160, 198]
    # 7 = [46, 66, 82, 102, 132, 166, 206]
    # 8 = [22, 31, 39, 47, 58, 75, 87, 107]
    # 7pyramid= [[1, 2, 3, 4, 5, 7], [3, 4, 5, 7, 9, 12], [5, 8, 11, 15, 20, 25], [12, 18, 23, 30, 40, 50], [24, 35, 45, 59, 80, 99]]
    # (which doesn't include the last one...), well it actually does. But it was scaled up for the last one. Interesting. So I really...
    # is the threshold I have too large?
