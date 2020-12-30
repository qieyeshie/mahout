package com.digquant;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ClassifyDonut4 {
    static class VectorHolder {
        Vector v;
        int target;
    }

    public static List<VectorHolder> loadData() throws IOException {
        // 1, 加载数据，构建特征向量值
        List<VectorHolder> trainDataList = new ArrayList<>();
        String path = "F:\\mahout\\day3\\day3\\data\\donut.csv";
        File file = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        // 1.1 去掉第一行标题
        String line = reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] lineArr = line.split(",");
            // 用x,y, a, b, c
            VectorHolder vHolder = new VectorHolder();
            vHolder.v = new DenseVector(6);
            double x = Double.parseDouble(lineArr[0]);
            double y = Double.parseDouble(lineArr[1]);
            double a = Double.parseDouble(lineArr[9]);
            double b = Double.parseDouble(lineArr[10]);
            double c = Double.parseDouble(lineArr[11]);
            vHolder.v.set(0, 1);
            vHolder.v.set(1, x);
            vHolder.v.set(2, y);
            vHolder.v.set(3, a);
            vHolder.v.set(4, b);
            vHolder.v.set(5, c);
            vHolder.target = Integer.parseInt(lineArr[3]) - 1;
            trainDataList.add(vHolder);
        }
        return trainDataList;
    }

    public static void main(String[] args) throws IOException {
        // 1，加载数据
        List<VectorHolder> trainDataList = loadData();

        // 2, 选择分类算法
        OnlineLogisticRegression lr = new OnlineLogisticRegression(2, 6, new L1());
        // 3 训练数据设置为30个
        List<VectorHolder> trainData2 = trainDataList.subList(0, 30);
        // 4 这里因为是使用递归下降的算法原因，需要对数据反复训练才能得到更好的结果
        // 4.1 这里循环训练100次
        for (int pass = 0; pass < 100; ++pass) {
            // 4.2 使用随机种子打算训练数据输入的顺序
            Random random = RandomUtils.getRandom();
            Collections.shuffle(trainData2, random);
            for (int i = 0; i < trainData2.size(); ++i) {
                VectorHolder vectorHolder = trainData2.get(i);
                // 3，训练算法
                lr.train(vectorHolder.target, vectorHolder.v);
            }
        }
        // 训练完毕后，使用剩下的数据集测试模型是否正确
        int correctNum = 0;
        for (int i = 30; i < trainDataList.size(); ++i) {
            VectorHolder vectorHolder = trainDataList.get(i);
            Vector preV = lr.classifyFull(vectorHolder.v);
            if (preV.maxValueIndex() == vectorHolder.target) {
                ++correctNum;
            }
        }

        System.out.printf("correctNum = %d correct precent = %f \n", correctNum, correctNum * 1.0 / (trainDataList.size() - 30));

    }


}
