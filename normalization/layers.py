import tensorflow as tf 

class ContextNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3, **kwargs):
        """
        Initialize the ContextNormalization layer.

        Parameters:
        - epsilon: A small positive value to prevent division by zero during normalization.
        """
        super(ContextNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        """
        Build the layer by creating sub-layers for learning mean and standard deviation.

        Parameters:
        - input_shape: The shape of the layer's input.

        This method initializes the layers for learning mean and standard deviation, based on the input shape.
        """
        self.input_dim = input_shape[0][-1]
        self.context_dim = input_shape[1][-1]

        # Layer for learning mean
        self.mean_layer = tf.keras.layers.Dense(
            units=self.input_dim,
            activation=None,
            kernel_initializer='glorot_uniform',
            trainable=True,
            input_shape=(self.context_dim,)
        )

        # Layer for learning standard deviation
        self.std_layer = tf.keras.layers.Dense(
            units=self.input_dim,
            activation=None,
            kernel_initializer='glorot_uniform',
            trainable=True,
            input_shape=(self.context_dim,)
        )

        super(ContextNormalization, self).build(input_shape)
        
        
    def config(self):
        config = super().get_config().copy()
        config.update({
        	"epsilon": self.epsilon
        })
        return config
        
        
    def call(self, inputs):
        """
        Apply the Context Normalization to the input data.

        Parameters:
        - inputs: A tuple of (x, context_id) where x is the data to be normalized, and context_id is the context identifier.

        Returns:
        - normalized_x: The normalized output data.
        """
        x, context_id = inputs

        # Calculate mean and standard deviation from context_id
        mean = self.mean_layer(context_id)
        std = self.std_layer(context_id)

        # Ensure standard deviation is positive
        std = tf.exp(std)

        # Determine the number of dimensions to expand
        num_expand_dims = len(x.shape) - 2

        # Expand mean and std dimensions accordingly
        for _ in range(num_expand_dims):
            mean = tf.expand_dims(mean, axis=1)
            std = tf.expand_dims(std, axis=1)

        # Perform normalization
        normalized_x = (x - mean) / (std + self.epsilon)

        return normalized_x
