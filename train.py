import os
import tensorflow as tf
import tensorflow.compat.v1 as tfc
from tensorflow.core.protobuf import saver_pb2


import driving_data
import model

LOGDIR = './save'

sess = tfc.InteractiveSession()

L2NormConst = 0.001

train_vars = tfc.trainable_variables()

loss = tf.reduce_mean(tfc.square(tfc.subtract(model.y_, model.y))) + tfc.add_n([tfc.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tfc.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tfc.initialize_all_variables())

# create a summary to monitor cost tensor
tfc.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op =  tfc.summary.merge_all()

saver = tfc.train.Saver(write_version = saver_pb2.SaverDef.V1)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tfc.summary.FileWriter(logs_path, graph=tfc.get_default_graph())

epochs = 30
batch_size = 100

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
