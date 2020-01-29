package com.example.ocrapplication;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.provider.MediaStore;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import android.media.ThumbnailUtils;
import android.util.DisplayMetrics;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.Canvas;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

public class MainActivity extends AppCompatActivity{

    private static final int pic_id = 100;

    Button camera_open_id;
    ImageView click_image_id;
    TextView text_view_id;
    Interpreter interpreter;

    private final List<String> OUTPUT_LABELS = Collections.unmodifiableList(
            Arrays.asList("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"));
    private static final ColorMatrix INVERT = new ColorMatrix(
            new float[]{
                    -1, 0, 0, 0, 255,
                    0, -1, 0, 0, 255,
                    0, 0, -1, 0, 255,
                    0, 0, 0, 1, 0
            });

    private static final ColorMatrix BLACKWHITE = new ColorMatrix(
            new float[]{
                    0.5f, 0.5f, 0.5f, 0, 0,
                    0.5f, 0.5f, 0.5f, 0, 0,
                    0.5f, 0.5f, 0.5f, 0, 0,
                    0, 0, 0, 1, 0,
                    -1, -1, -1, 0, 1
            });

    private void loadMnistClassifier() {
        try {
            ByteBuffer byteBuffer = loadMainFile();
            interpreter = new Interpreter(byteBuffer);
        } catch (IOException e) {
            Toast.makeText(this, "MNIST model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera_open_id = (Button)findViewById(R.id.camera_button);
        click_image_id = (ImageView)findViewById(R.id.click_image);
        text_view_id = (TextView)findViewById(R.id.textView);

        loadMnistClassifier();

        camera_open_id.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent camera_intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(camera_intent, pic_id);
            }
        });
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == pic_id) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            click_image_id.setImageBitmap(photo);

            Bitmap squareBitmap = ThumbnailUtils.extractThumbnail(photo, getScreenWidth(), getScreenWidth());
            Bitmap preprocessedImage = prepareImageForClassification(squareBitmap);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(preprocessedImage);

            float[][] result = new float[1][10];
            interpreter.run(byteBuffer, result);

            List<Classification> recognitions = getSortedResult(result);
            text_view_id.setText(recognitions.toString());
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(3136);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[28 * 28];

        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int pixel : pixels) {
            float rChannel = (pixel >> 16) & 0xFF;
            float gChannel = (pixel >> 8) & 0xFF;
            float bChannel = (pixel) & 0xFF;
            float pixelValue = (rChannel + gChannel + bChannel) / 3 / 255.f;

            byteBuffer.putFloat(pixelValue);
        }

        return byteBuffer;
    }

    private List<Classification> getSortedResult(float[][] resultsArray) {
        PriorityQueue<Classification> sortedResults = new PriorityQueue<>(
                3,
                (lhs, rhs) -> Float.compare(rhs.confidence, lhs.confidence)
        );

        for (int i = 0; i < 10; ++i) {
            float confidence = resultsArray[0][i];
            if (confidence > 0.1f) {
                sortedResults.add(new Classification(OUTPUT_LABELS.get(i), confidence));
            }
        }

        return new ArrayList<>(sortedResults);
    }

    private int getScreenWidth() {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);

        return displayMetrics.widthPixels;
    }

    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0);
        colorMatrix.postConcat(BLACKWHITE);
        colorMatrix.postConcat(INVERT);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(colorMatrix);

        Paint paint = new Paint();
        paint.setColorFilter(f);

        Bitmap bmpGrayscale = Bitmap.createScaledBitmap(
                bitmap,
                28,
                28,
                false);
        Canvas canvas = new Canvas(bmpGrayscale);
        canvas.drawBitmap(bmpGrayscale, 0, 0, paint);
        return bmpGrayscale;
    }

    private MappedByteBuffer loadMainFile() throws IOException{
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mnist_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
}
