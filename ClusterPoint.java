package com.digquant;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class ClusterPoint {


    /**
     * 加载数据
     *
     * @return
     */
    static public List<Vector> loadData() {
        // 使用数据构建
        double[][] points = {
                {1, 1}, {2, 1}, {1, 2},
                {2, 2}, {3, 3}, {8, 8},
                {9, 8}, {8, 9}, {9, 9}
        };
        //
        List<Vector> vectorList = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            double[] fr = points[i];
            Vector vec = new RandomAccessSparseVector(fr.length);
            vec.assign(fr);
            vectorList.add(vec);
        }
        return vectorList;
    }

    public static void main(String[] args) {
        // 1，加载数据
        List<Vector> vectors = loadData();
        // 2. 设置初始的簇中心，并设置计算点到簇中心距离的算法
        List<Cluster> clusterModels = new ArrayList<>();
        for (int i = 0; i < 2; ++i) {
            Vector vec = vectors.get(i);
            clusterModels.add(new Kluster(vec, i, new EuclideanDistanceMeasure()));
        }
        // 3，构建聚类算法
        KMeansClusteringPolicy kMeansClusteringPolicy = new KMeansClusteringPolicy();
        ClusterClassifier classifier = new ClusterClassifier(clusterModels, kMeansClusteringPolicy);
        kMeansClusteringPolicy.update(classifier);
        // 4， 运行算法，这里默认迭代10次
        for (int i = 0; i < 10; ++i) {
            for (Vector vector : vectors) {
                Vector probabilities = classifier.classify(vector);
                Vector weights = kMeansClusteringPolicy.select(probabilities);
                for (Vector.Element e : weights.nonZeroes()) {
                    int index = e.index();
                    classifier.train(index, vector, weights.get(index));
                }
            }
            // 4.1 每次簇中心迭代完毕，计算先选出来簇中心的变化大小
            classifier.close();
            classifier.getPolicy().update(classifier);
            List<Cluster> models = classifier.getModels();
            boolean bBreak = false;
            // 4.2 当簇中心的变化不大时，推出计算
            for (Cluster cluster:  models) {
                if (cluster.isConverged()) {
                    bBreak = true;
                    break;
                }
            }
            if (bBreak) {
                break;
            }
        }
        // 5，根据计算出来的簇中心，计算各个点到各个簇中心的距离，然后和离自己最新的簇中心
        //    成为一簇
        for (Vector vector : vectors) {
            Vector pdfPerCluster = classifier.classify(vector);
            if (pdfPerCluster.maxValue() >= 0) {
                System.out.printf("clusterId = %d, point(%f, %f)\n", pdfPerCluster.maxValueIndex(), vector.get(0),vector.get(1));
            }
        }
    }
}
