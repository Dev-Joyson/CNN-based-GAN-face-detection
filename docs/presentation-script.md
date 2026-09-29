# Presentation script: slides 10 to 23 (Results & Discussion)

Spoken script for the Canva deck "V3 Final Presentation", from the Results & Discussion
divider up to the Demonstration divider (slide 24). About 9 minutes at a normal pace.
Numbers match the slides. Point at the tables, don't read the cells out loud.

Naming on the slides: base model (five conv blocks, 1.01M), stride-2 model (same, first
layer stride 2), final model (stride 2 plus half width, 262k). Technical words stay as
they are; each one gets a plain-words explanation the first time it comes up.

---

## Slide 10: Results & Discussion (divider)

That was the setup. Now what happened when we ran it. The results didn't come in a
straight line. There were six steps, and each one changed what we did next. I'll go
through them in order.

---

## Slide 11: Our dual branch model on resized faces reached 0.90 accuracy

We started with the design from the proposal. A small CNN with two branches. One branch
looks at the pixels. The other looks at the frequency spectrum, which you get from an
FFT, a transform that shows how much fine detail and how much coarse detail an image
has. We join the two branches at the end.

The images were full size, 1024 by 1024, from FFHQ for the real faces and StyleGAN2 for
the fakes. We resized them down to 256 before feeding them in. That's what everybody
does.

On the test set we got 0.90 accuracy. Nine out of ten right. Not bad for a start. But
the pretrained detectors we compare against were already in the high nineties. So the
question was: how do we close that gap?

---

## Slide 12: Learning from a bigger teacher made our model worse

Our first idea was knowledge distillation. In simple terms: take a big model that already
does this well, and let our small model learn from its answers, instead of only from the
plain real-or-fake labels. It's a well known way to help small models.

We tried two different teachers. Both times our model came out worse. 0.90 dropped to
0.84 with one teacher and 0.80 with the other.

So better labels weren't the answer. That made us stop and think. If the model can't
learn more from a smarter teacher, maybe the problem isn't the teaching. Maybe it's what
we're feeding in.

---

## Slide 13: Resizing was deleting the GAN fingerprint

And it was. This is the finding the whole project turns on.

A GAN leaves a fingerprint in the image. It's a very fine pattern, at the level of single
pixels, that comes from how the generator builds the image up from small to big. When you
resize a 1024 image down to 256, you average blocks of pixels together, and that pattern
gets smoothed away. The model never gets to see it.

So we stopped resizing. We take the middle 512 by 512 of the original image, that's the
blue box, and during training we cut random 256 by 256 windows out of it, the red box.
Every pixel the network sees is an original pixel. Nothing is resampled. We call these
native crops.

Same model, same training. Accuracy went from 0.90 to 0.99. That one change was worth
more than everything else we tried put together.

---

## Slide 14: Native crops kept climbing where resized input plateaued

You can see it in the training curves too. The red line is the resized input. It goes up
early and then just sits there. There's nothing more for the model to find.

The blue line is native crops. It keeps climbing, slowly, for a very long time, well past
a hundred epochs. An epoch is one pass over all the training images. The information is
there, it just takes the model a while to lock onto it. The orange and green lines are
variants I'll get to in a moment. The point is they follow the same shape.

Now, once the crops were working, we had to ask about the FFT branch. It was designed for
the resized pipeline. Was it still doing anything?

---

## Slide 15: The FFT branch only helped when the image was resized

Short answer, no. We trained with and without it on both pipelines.

On resized input, the FFT branch was worth about two points, 0.88 to 0.90. That makes
sense. When the pixel branch has lost the fingerprint, the spectrum is a second way to
catch it.

On native crops, both versions land at 0.99. The pixel branch can already see the
fingerprint, so the spectrum adds nothing.

So we dropped it. That leaves a plain five-block CNN with about one million parameters.
Parameters are the numbers the network learns, so that's a measure of its size. And now
it was time to compare against the pretrained models.

---

## Slide 16: Base model: 6× faster than MobileNet on GPU, but 16× in compute

Here's that comparison against MobileNetV3-Small, the closest pretrained model in size.
Both are about one million parameters.

On the GPU, ours runs in about one millisecond per image. MobileNet takes over six. That's
because our network is just plain convolutions, and GPUs are very good at those.

But look at the compute row. Ours needs 1.11 G MACs per image. A MAC is one multiply and
one add, the basic step inside a convolution, and G means a billion. So 1.11 billion of
those per image. MobileNet needs 0.07. That's sixteen times more arithmetic for us. And
on a single CPU core, where arithmetic is what you pay for, MobileNet is nearly ten times
faster: 3 milliseconds against 30.

So we had a model that was accurate and fast on a GPU, but heavy everywhere else. For a
project whose whole point is being lightweight, that wasn't good enough. We had to cut
the compute.

---

## Slide 17: A stride-2 first layer makes the model faster

Our first change was a small one. The first conv layer looks at the whole 256 by 256
image, so it does a lot of the work. We made it move two pixels at a time instead of
one. That's what stride 2 means. Only this first layer changes. The other four stay as
they are, but because the first one now gives out a smaller picture, every layer after
it has a quarter of the work to do.

That's the second point: a quarter of the arithmetic. The table says it in MACs, which
is just the number of multiply-adds for one image: 1.11 billion down to 0.28. On one
CPU core, that's 30 milliseconds down to under 8.

What does it cost? Half a point of accuracy, 0.99 to 0.984. And the number of
parameters is exactly the same, 1.01 million, because stride only changes how the
filters move, not how many there are.

That's better, but MobileNet still needs less arithmetic per image than this. So the
next step was to cut the width in half.

---

## Slide 18: 4× smaller than MobileNet, and faster on one CPU core

This is the final model. On top of stride 2, we made every layer half as wide. Width
means how many filters a layer has, and each filter is one pattern the layer looks for.
So 16, 32, 64, 128, 128 filters instead of double that. Fewer filters, fewer
parameters, and less arithmetic in every layer.

That brings the arithmetic down to 0.073 G MACs per image. That's the same as MobileNet,
with a quarter of its parameters. The chart shows the whole journey. The two light bars
on the left are the dual-branch model, resized and then on crops. The three purple bars
are the base model, the stride-2 model, and the final one. The red dashed line is
MobileNet, and the last bar sits right on it.

It cost us about two accuracy points, 0.99 down to 0.98. And it was slow to train, more
than two hundred epochs. But when it runs, this is now the lightest model in the room.

So at 0.98, what do the mistakes actually look like?

---

## Slide 19: The final model gets 98 out of 100 test images right

The test set is 7,500 images the model never saw during training. 3,750 real faces from
FFHQ, 3,750 fakes from StyleGAN2.

The model gives each image a score between 0 and 1 for "fake". We call it fake above
0.5. At that cut-off, 45 real faces were called fake, and 106 fakes got through as real.
So it's a bit more likely to let a fake through than to doubt a real photo, and both
numbers are small. That's the 0.98.

We didn't tune that 0.5 on the test set. It's just the middle. And this is one training
run, one seed.

So it works. But which parts of the face is it actually using to decide?

---

## Slide 20: Where the final model looks before it decides

This is Grad-CAM. It's a way of colouring the image by how much each part pushed the
model's decision. Warm colours mean "this part mattered". We take it from the last conv
layer. Top row is four real faces, bottom row four fakes, and all eight were classified
correctly.

One thing to be clear about. This shows where the evidence was, not what the model
"thinks". And what you see is that it isn't checking eyes or mouths. The hot patches sit
on skin texture, which is where the generator's fingerprint lives. That fits the whole
story since slide 13. The model learned a texture-level clue, not a face-level one.

The third fake, with the thinnest colour, is also the least sure of the eight, at 0.69.
That fits too.

So how does it stack up against the other models?

---

## Slide 21: Our final model is the smallest and fastest, and 2% less accurate

Everything in this table was measured in one sitting. Same L4 GPU, same Xeon CPU limited
to one core, fp32, which means normal full precision, one image at a time. That matters,
and I'll come back to why.

Our three are the top rows, in the order we built them. Base model, stride-2 model, final
model. Then the three pretrained ones. The bold number in each column is the best.

Final model: 262 thousand parameters, about one millisecond on the GPU, 2.3 milliseconds
on one CPU core, 0.98 accuracy. Against MobileNet, the closest one: four times fewer
parameters, six times faster on the GPU, 1.4 times faster on one CPU core.

And the price is accuracy. They're all at 0.998 or above. We're at 0.98. Two points is
real and we don't hide it. But this is a model trained from nothing, on one dataset, with
no pretraining, at a quarter the size of the smallest model it's compared to.

---

## Slide 22: On one CPU core, our final model is the fastest of the six

Same numbers as a picture, and this time the axis is one CPU core, no GPU. That's the
kind of setting a phone or a small device runs in. Time per image along the bottom, on a
log scale, so each step to the right is ten times slower. Accuracy up the side. The size
of each circle is the number of parameters.

Read it left to right. Our final model is the leftmost circle. MobileNet is just behind
it. Then the stride-2 model, then EfficientNet, and the base model sits next to it at
about 30 milliseconds. Xception is the big one out at 128.

So two changes took our model from 30 milliseconds to 2.3 on a single core, the fastest
of the six, and it still runs in about one millisecond on the GPU. The pretrained ones
are higher on the accuracy axis. That's the trade-off in one picture.

Before the demo, two things we got wrong along the way, because I think they matter.

---

## Slide 23: Two corrections we made along the way

First, memory. Early on we reported that our base model used 1.3 gigabytes of GPU memory,
which looked terrible next to MobileNet's 150 megabytes. That number turned out to be
scratch space the GPU library grabs on the very first call while it tries out different
ways of running the model. After a warm-up, the real figure is 171 megabytes. We
re-measured every model the same way.

Second, CPU time. Our first CPU numbers for MobileNet went from 1.3 milliseconds to 6.6
depending on which Colab machine we happened to get. Different chips, different
instruction sets. So every number in that table comes from one machine, in one sitting,
and the machine is written down next to the results.

I mention these because a resource comparison is only as good as its measurement, and
we'd rather show you the correction than the wrong number.

That's the results. Now the final model live, next to MobileNet, on the same images.

---

*Next: slide 24, Demonstration. The demo page scores both models on every image and
shows both latencies; say once that laptop latency is for feel and the measured numbers
are on slide 21.*
