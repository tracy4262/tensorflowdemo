"""
 * Created with PyCharm.
 * User: 彭诗杰
 * Date: 2018/7/4
 * Time: 21:31
 * Description: 测试tensorflow环境是否安装成功
"""
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
