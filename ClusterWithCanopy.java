package com.digquant;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyClusterer;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.display.DisplayClustering;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;


import java.awt.*;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.List;

/**
 * 继承了Mahout的聚类可视化类
 */
public class ClusterWithCanopy extends DisplayClustering {
    /**
     * 默认构造函数
     */
    public ClusterWithCanopy() {
        initialize();
        /**
         * 设置对话框的名字
         */
        this.setTitle("case 1");
    }

    /**
     * 所有点的数据
     */
    static private List<Vector> sampleData2;
    static private ClusterClassifier classifier;

    public static void main(String[] args){
        // 1, 随机生成  400个 x,y坐标轴的平均值为1，偏差大小为3点
        List<Vector> sampleData = new ArrayList<>();
        RandomPointsUtil.generateSamples(sampleData, 400, 1, 1, 3);
        // 2, 先保存生成的点，因为canopy会删除点
        sampleData2 = new ArrayList<>();
        sampleData2.addAll(sampleData);
        // 3, 使用 canopy生成K-means的初始簇中心，设置T1 = 3.0 T2 = 1.5
        List<Canopy> canopies = CanopyClusterer.createCanopies(sampleData, new EuclideanDistanceMeasure(), 3.0, 1.5);
        // 4， 使用 canopy生成的簇中心作为kmeans的初始簇重新
        List<Cluster> clusterModels = new ArrayList<>();
        int clusterId = 0;
        for (Canopy canopy : canopies) {
            Vector vec = canopy.getCenter();
            clusterModels.add(new Kluster(vec, clusterId++, new EuclideanDistanceMeasure()));
        }

        // 3，构建聚类算法
        KMeansClusteringPolicy kMeansClusteringPolicy = new KMeansClusteringPolicy();
        classifier = new ClusterClassifier(clusterModels, kMeansClusteringPolicy);
        kMeansClusteringPolicy.update(classifier);
        // 4， 运行算法，这里默认迭代10次
        for (int i = 0; i < 100; ++i) {
            for (Vector vector : sampleData2) {
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
            for (Cluster cluster : models) {
                if (cluster.isConverged()) {
                    bBreak = true;
                    break;
                }
            }
            if (bBreak) {
                break;
            }
        }

        new ClusterWithCanopy();
    }

    /**
     * 画点的
     * @param g2
     */
    protected void plotSampleData2(Graphics2D g2) {
        double sx = (double) res / DS;
        g2.setTransform(AffineTransform.getScaleInstance(sx, sx));
        g2.setColor(Color.BLACK);
        Vector dv = new DenseVector(2).assign(SIZE / 2.0);
        plotRectangle(g2, new DenseVector(2).assign(2), dv);
        plotRectangle(g2, new DenseVector(2).assign(-2), dv);

        g2.setColor(Color.DARK_GRAY);
        dv.assign(0.03);
        for (Vector v : sampleData2) {
            plotRectangle(g2, v, dv);
        }
    }

    final Color[] COLORS_My = {Color.red, Color.orange, Color.yellow, Color.green, Color.blue, Color.magenta,
            Color.lightGray};

    /**
     * 画红色圈的
     * @param g2
     */
    void plotClusters2(Graphics2D g2) {
        List<Cluster> models = classifier.getModels();

        g2.setStroke(new BasicStroke(3));
        g2.setColor(COLORS_My[Math.min(COLORS_My.length - 1, 0)]);
        for (Cluster cluster : models) {
//            plotEllipse(g2, cluster.getCenter(), cluster.getRadius().times(3));
            plotEllipse(g2, cluster.getCenter(), cluster.getRadius());
        }
    }

    @Override
    public void paint(Graphics g) {
        plotSampleData2((Graphics2D) g);
        plotClusters2((Graphics2D) g);
    }
}
