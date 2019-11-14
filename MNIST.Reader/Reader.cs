using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST.Reader
{
    public class Reader
    {
		public MNISTData Read(string imagesPath, string labelsPath)
		{
			// pixels
			BinaryReader reader = new BinaryReader(File.OpenRead(imagesPath));

			int magic = NextInt32(reader);
			int size = NextInt32(reader);
			int rows = NextInt32(reader);
			int cols = NextInt32(reader);

			byte[] pixels = new byte[size * rows * cols];
			reader.Read(pixels, 0, pixels.Length);

			reader.Close();
			reader.Dispose();

			// labels
			reader = new BinaryReader(File.OpenRead(labelsPath));

			int l_magic = NextInt32(reader);
			int l_size = NextInt32(reader);

			byte[] labels = new byte[size];
			reader.Read(labels, 0, l_size);

			MNISTData data = new MNISTData(size, cols, rows);
			data.SetData(pixels, labels);
			return data;
		}

		private int NextInt32(BinaryReader reader)
		{
			byte[] buffer = reader.ReadBytes(4);
			if (BitConverter.IsLittleEndian)
				Array.Reverse(buffer);
			return BitConverter.ToInt32(buffer, 0);
		}
    }
}
