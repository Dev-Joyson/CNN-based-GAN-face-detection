# Presentation script: slides 10 to 23 (Results & Discussion)

Spoken script for the Canva deck "V3 Final Presentation", from the Results & Discussion
divider up to the Demonstration divider (slide 24). Roughly 8 to 9 minutes at a normal
speaking pace. Numbers match the slides exactly; do not read the table cells aloud, point
at them.

---

## Slide 10: Results & Discussion (divider)

So that was the setup. Now let me walk you through what actually happened when we ran
it, because the results didn't come in a straight line. We went through six steps, and
each one changed what we did next. I'll take them in order.

---

## Slide 11: Our dual branch model on resized faces reached 0.90 accuracy

We started with the design from the proposal: a small CNN with two branches. One branch
looks at the pixels, the other looks at the frequency spectrum from an FFT, and we
combine them at the end.

The images were the full 1024 by 1024 faces from FFHQ and StyleGAN2, and we resized
them down to 256 before feeding them in, which is the normal thing to do.

On the test set that gave us 0.90 accuracy. Nine out of ten right. That's a decent
start, but the pretrained detectors we were comparing against were already in the high
nineties. So the obvious question was: how do we close that gap?

---

## Slide 12: Learning from a bigger teacher made our model worse

Our first idea was knowledge distillation. Take a big pretrained model that already does
this well, and let our small model learn from its outputs instead of just the hard
real-or-fake labels. It's a well-known trick for small models.

We tried two different teachers. Both students came out worse than the plain model.
0.90 dropped to 0.84 with one teacher and 0.80 with the other.

So better labels weren't the answer. And that made us stop and think. If the model
can't learn more from a smarter teacher, maybe the problem isn't the training signal.
Maybe it's what we're feeding in.

---

## Slide 13: Resizing was deleting the GAN fingerprint

And it was. This is the finding the whole project turns on.

A GAN leaves a fingerprint in the image. It's a very fine, pixel-level pattern that
comes from how the generator upsamples. When you resize a 1024 image down to 256, you
average over blocks of pixels, and that pattern is smoothed away. The model never gets
to see it.

So we stopped resizing. We take the centre 512 by 512 region of the original image,
that's the blue box, and during training we cut random 256 by 256 windows out of it,
the red box. Every pixel the network sees is an original pixel. Nothing is resampled.

Same model, same training recipe. Accuracy went from 0.90 to 0.99. That one change was
worth more than everything else we tried combined.

---

## Slide 14: Native crops kept climbing where resized input plateaued

You can see it in the training curves too. The red line is the resized input. It goes
up early and then just sits there. There's nothing more for the model to find.

The blue line is native crops. It keeps climbing, slowly, for a very long time, well
past a hundred epochs. The information is there, it just takes the model a while to
lock onto it. The orange and green lines are variants I'll get to in a moment; the point
is that they follow the same shape.

Now, once the crops were working, we had to ask about the FFT branch. It was designed
for the resized pipeline. Was it still doing anything?

---

## Slide 15: The FFT branch only helped when the image was resized

Short answer, no. We trained with and without it on both pipelines.

On resized input, the FFT branch was worth about two points, 0.88 to 0.90. That makes
sense: when the spatial branch has lost the fingerprint, the spectrum is a second way
to catch it.

On native crops, both versions land at 0.99. The spatial branch can already see the
fingerprint, so the spectrum adds nothing.

So we dropped it. That leaves us with a plain five-block CNN with about one million
parameters, which is a cleaner story anyway. And now it was time to actually compare
against the pretrained models.

---

## Slide 16: Full-size model: 6× faster than MobileNet on GPU, but 16× in compute

Here's that comparison, against MobileNetV3-Small, which is the closest pretrained
model in size. Both are about one million parameters.

On the GPU, ours runs in about one millisecond per image, MobileNet takes over six.
That's because our network is just plain convolutions, and GPUs are very good at those.

But look at the compute column. Ours needs 1.11 billion multiply-adds per image,
MobileNet needs 0.07. That's sixteen times more arithmetic. And on a single CPU core,
where arithmetic is what you pay for, MobileNet is nearly ten times faster: 3 milliseconds
versus 30.

So we had a model that was accurate and fast on a GPU, but heavy everywhere else. For a
project whose whole point is being lightweight, that wasn't good enough. We had to cut
the compute.

---

## Slide 17: A stride-2 first layer makes the model faster

The first cut was simple. The very first convolution layer works at full resolution,
and that layer alone is a big share of the cost. We gave it a stride of two, so it
steps two pixels at a time and every layer after it works on a quarter of the area.

Compute dropped four times, from 1.11 to 0.28 billion multiply-adds. CPU time went
from 30 milliseconds to under 8. Accuracy went from 0.99 to 0.984, so we lost about
half a point. The parameter count doesn't change, because stride doesn't touch the
weights.

That's a good trade, but we were still at four times MobileNet's compute and two and
a half times slower on one core. So we went one step further.

---

## Slide 18: Final model: 1/4 of MobileNet's size and faster on a single CPU core

The second cut was width. We halved the number of channels in every layer. Instead of
32, 64, 128, 256, 256, it's now 16, 32, 64, 128, 128.

This chart is the whole journey in one picture. We started at 1.2 billion multiply-adds
and 0.90 accuracy with the resized dual-branch model. Crops took us to 0.99 at the same
cost. Dropping the FFT trimmed it slightly. Stride two brought it down to 0.28. Half
width brings it to 0.073 billion, and that's the final model.

What did it cost us? About two accuracy points, 0.99 down to 0.98. And it's slow to
train, it took over two hundred epochs to converge. But at inference it's now smaller
than MobileNet in both parameters and compute.

So the natural question is: at 0.98, what do those errors actually look like?

---

## Slide 19: Final model on 7,500 test images

The test set is 7,500 images the model has never seen, half real, half StyleGAN2.

At a threshold of 0.5, it gets 0.98. Out of 3,750 real faces, 45 were called fake.
Out of 3,750 fakes, 106 slipped through. So the model is slightly more likely to miss
a fake than to accuse a real photo, and both error types are small.

I want to be clear that this is one seed and one threshold. We didn't tune the
threshold on the test set.

Now, we can count the errors, but we also wanted to know what the model is actually
looking at when it decides.

---

## Slide 20: What the final model looks at in a 256² crop

This is Grad-CAM on the last convolution layer. The warm colours show which parts of
the crop pushed the decision. Top row is four real faces, bottom row is four fakes,
and all eight are classified correctly.

The thing to notice is that it isn't looking at anything semantic. It's not checking
whether the eyes match or the teeth look right. The hot spots sit on skin, on hair, on
texture, on the regions where the generator's upsampling pattern is strongest. That's
consistent with the fingerprint story: the model learned a texture-level statistic,
not a face-level one.

On the fakes, the attention is broad and confident. On the reals, it's more scattered,
which fits: a real photo has no fingerprint to find, so the model is confirming an
absence.

So it works, and we have a reasonable idea why. Let's put it next to everything else
we measured.

---

## Slide 21: Our final model is the smallest and fastest, for about 2 accuracy points

Everything in this table was measured in one session, on the same L4 GPU and the same
Xeon CPU, at batch size one, in fp32. That matters, and I'll come back to why.

Final model: 262 thousand parameters, about one millisecond on the GPU, 2.3
milliseconds on a single CPU core, 0.98 accuracy.

Against MobileNetV3-Small, the closest competitor: a quarter of the parameters, six
times faster on the GPU, and 1.4 times faster on one CPU core. Against EfficientNet
and Xception, the gap is much larger on every resource column.

And the price is accuracy. The pretrained models are all at 0.998 or above. We're at
0.98. Two points is real, and we don't hide it. But this is a model trained from
scratch, on one dataset, with no pretraining, at a quarter of the size of the smallest
thing it's compared against.

---

## Slide 22: On the GPU ours run in about 1 ms, pretrained models take 5.5–9.5 ms

Same numbers, as a picture. Latency along the bottom, accuracy up the side, and the
size of each circle is the number of parameters.

Our three models are the small circles on the left, all at about one millisecond. The
pretrained models sit between five and nine and a half milliseconds, and they're higher
on the accuracy axis. Xception is the big circle: twenty-one million parameters for
half a point over our full-size model.

This is the trade-off in one glance. You can have the last point of accuracy, or you
can have a model that costs almost nothing to run. We chose the second one, and the
gap is small enough to be honest about.

Before the demo, I want to mention two things we got wrong along the way, because I
think they matter.

---

## Slide 23: Two corrections we made along the way

First, memory. Early on we reported that our full-size model used 1.3 gigabytes of GPU
memory, which looked terrible next to MobileNet's 150 megabytes. It turned out that
number was cuDNN's autotuning scratch space on the first call. After a warm-up, the
real figure is 171 megabytes. We re-measured every model the same way.

Second, CPU latency. Our first CPU numbers for MobileNet varied from 1.3 milliseconds
to 6.6 depending on which Colab machine we happened to land on. Different CPUs, different
instruction sets. So every number you saw in that table comes from one machine, in one
session, and the machine is recorded alongside the results.

I mention these because a resource comparison is only as good as its measurement, and
we'd rather show you the correction than the wrong number.

That's the results. Now let me show you the final model running live.

---

*Next: slide 24, Demonstration.*
