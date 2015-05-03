import math
import numpy as np


class OtsuPyramid(object):
    """ class segments histogram into pyramid of histograms, each half the size of the previous. Generating omega
    and mu values allow for this to work extremely fast
    """

    def load_image(self, im, bins=256):
        self.im = im
        # bins = number of intensity levels
        hist, ranges = np.histogram(im, bins)  # this also works exaclty the same
##        hist = cv2.calcHist([im], [0], None, [bins], [0, bins])
        hist = [int(h) for h in hist]  # so we convert the numpy array to list of ints
        histPyr, omegaPyr, muPyr, ratioPyr = self._create_histogram_and_stats_pyramids(hist)
        self.omegaPyramid = [omegas for omegas in reversed(omegaPyr)]  # so that pyramid[0] is the smallest pyramid
        self.muPyramid = [mus for mus in reversed(muPyr)]
        self.ratioPyramid = ratioPyr
        
    def _create_histogram_and_stats_pyramids(self, hist):
        """ expects hist to be a single list of numbers (no numpy array)
        """
        bins = len(hist)
        ratio = 2  # eventually you can replace this with a list if you cannot evenly compress a histogram
        reductions = int(math.log(bins, ratio))
        compressionFactor = []
        histPyramid = []
        omegaPyramid = []
        muPyramid = []
        for i in range(reductions):
            histPyramid.append(hist)
            reducedHist = [sum(hist[i:i+ratio]) for i in range(0, bins, ratio)]
            hist = reducedHist  # collapse a list to half its size, combining the two collpased numbers into one
            bins = bins / ratio  # update bins to reflect the length of the new histogram
            compressionFactor.append(ratio)
        compressionFactor[0] = 1  # first "compression" was 1, aka it's the original histogram
        for hist in histPyramid:
            omegas, mus, muT = self._calculate_omegas_and_mus_from_histogram(hist)
            omegaPyramid.append(omegas)
            muPyramid.append(mus)
        return histPyramid, omegaPyramid, muPyramid, compressionFactor

    def _calculate_omegas_and_mus_from_histogram(self, hist):
        probabilityLevels, meanLevels = self._calculate_histogram_pixel_stats(hist)
        bins = len(probabilityLevels)
        ptotal = np.longdouble(0)  # these numbers are critical towards calculations, so we make sure they are big precision
        omegas = []  # sum of probability levels up to k
        for i in range(bins):
            ptotal += probabilityLevels[i]
            omegas.append(ptotal)
        mtotal = np.longdouble(0)
        mus = []
        for i in range(bins):
            mtotal += meanLevels[i]
            mus.append(mtotal)
        muT = float(mtotal)  # muT is the total mean levels.
        return omegas, mus, muT

    def _calculate_histogram_pixel_stats(self, hist):
        bins = len(hist)  # bins = number of intensity levels
        N = np.longdouble(sum(hist))  # N = number of pixels in image. Make it float so that division by N will be a float
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
            scaling = self.ratioPyramid[i]  # how much our "just analyzed" pyramid was compressed from the previous one
            deviate = scaling  #scaling  # deviate should be equal to the compression factor of the previous histogram.
            thresholds = [t * scaling for t in thresholds]  # list(np.array(thresholds) * 2)
        return [t / scaling for t in thresholds]  # return readjusted threshold (since it was scaled up incorrectly in last loop)
    
    def _get_starting_pyramid_index(self, k):
        """ given the number of thresholds, return the minium starting index
        of pyramid to use in calculating thresholds
        """
        for i, pyramid in enumerate(self.omegaPyramid):
            if len(pyramid) >= k:
                return i

    def apply_thresholds_to_image(self, thresholds, im=None):
        if im is None:
            im = self.im
        k = len(thresholds)
        bookendedThresholds = [None] + thresholds + [None]
        greyValues = [0] + [int(256 / k * (i + 1)) for i in range(0, k - 1)] + [255]  # I think you need to use 255 / k *...
        finalImage = np.zeros(im.shape, dtype=np.uint8)
        for i in range(k + 1):
            kSmall = bookendedThresholds[i]
            bw = np.ones(im.shape, dtype=np.bool8)  # True portions of bw represents pixels between the two thresholds
            if kSmall:
                bw = (im >= kSmall)
            kLarge = bookendedThresholds[i + 1]
            if kLarge:
                bw &= (im < kLarge)
            greyLevel = greyValues[i]
            greyImage = bw * greyLevel  # apply grey-color to black-and-white image
            finalImage += greyImage  # add grey portion to image. There should be no overlap between each greyImage added
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
        for thresholds in self._jitter_thresholds_generator(originalThresholds, 0, self.bins):
            variance = self.sigmaB.get_total_variance(thresholds)
            newResult = (variance, thresholds)
            bestResults = max(bestResults, newResult)
        return bestResults[1]

    def _jitter_thresholds_generator(self, thresholds, min_, max_):
        pastThresh = thresholds[0]
        if len(thresholds) == 1:
            for offset in range(-self.deviate, self.deviate + 1):  # -2 through +2
                thresh = pastThresh + offset
                if thresh < min_ or thresh >= max_:
                    continue  # skip since we are conflicting with bounds
                yield [thresh]
        else:
            thresholds = thresholds[1:]  # new threshold without our threshold included
            m = len(thresholds)  # number of threshold left to generate in chain
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                if thresh < min_ or thresh + m >= max_:  # verify we don't use the same value as the previous threshold
                    continue                         # and also verify our current threshold will not push the last threshold past max
                recursiveGenerator = self._jitter_thresholds_generator(thresholds, thresh + 1, max_)
                for otherThresholds in recursiveGenerator:
                    yield [thresh] + otherThresholds


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


# my method is faster than this 1999 approach
# http://www.iis.sinica.edu.tw/JISE/2001/200109_01.pdf
# possibly a similar method
# http://www-cs.engr.ccny.cuny.edu/~wolberg/cs470/doc/Otsu-KMeansHIS09.pdf
# I think my method has not been used before. It took ~ 5 min but mine even computed 8 threshold levels
# I have changed the deviate variable from 2 to 256, and the results are always the same thresholds
# but it should work with just deviate set to 2
if __name__ == '__main__':
    import cv2
    filename = 'tractor.png'
    dot = filename.index('.')
    prefix, extension = filename[:dot], filename[dot:]
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    otsu = OtsuFastMultithreshold()
    otsu.load_image(im)
    for k in [1, 2, 3, 4, 5, 6]:
        kThresholds = otsu.calculate_k_thresholds(k)
        print(kThresholds)
        crushed = otsu.apply_thresholds_to_image(kThresholds)
        cv2.imwrite(prefix + '_crushed_' + str(len(kThresholds)) + extension, crushed)
