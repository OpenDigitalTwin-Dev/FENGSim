#include "SPCMainWindow.h"
#include "ui_SPCMainWindow.h"

SPCMainWindow::SPCMainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SPCMainWindow)
{
    ui->setupUi(this);

    m_bw = 100;                 // initial bandwidth, will be scaled by 100 so 1.0
    m_kernel = 0;               // initial kernel: gaussian
    m_cmd = "c(rnorm(100,0,1), rnorm(50,5,1))"; // simple mixture
    m_R["bw"] = m_bw;           // pass bandwidth to R, and have R compute a temp.file name
    m_tempfile = QString::fromStdString(Rcpp::as<std::string>(m_R.parseEval("tfile <- tempfile()")));
    m_svgfile = QString::fromStdString(Rcpp::as<std::string>(m_R.parseEval("sfile <- tempfile()")));
    setupDisplay();
}

SPCMainWindow::~SPCMainWindow()
{
    delete ui;
}


void SPCMainWindow::setupDisplay(void)  {


    m_svg = new QSvgWidget();
    setCentralWidget(m_svg);
    runRandomDataCmd();         // also calls plot()


}

void SPCMainWindow::plot(void) {
    const char *kernelstrings[] = { "gaussian", "epanechnikov", "rectangular", "triangular", "cosine" };
    m_R["bw"] = m_bw;
    m_R["kernel"] = kernelstrings[m_kernel]; // that passes the string to R
    std::string cmd0 = "svg(width=6,height=6,pointsize=10,filename=tfile); ";
    std::string cmd1 = "plot(density(y, bw=bw/100, kernel=kernel), xlim=range(y)+c(-2,2), main=\"Kernel: ";
    std::string cmd2 = "\"); points(y, rep(0, length(y)), pch=16, col=rgb(0,0,0,1/4));  dev.off()";
    std::string cmd = cmd0 + cmd1 + kernelstrings[m_kernel] + cmd2; // stick the selected kernel in the middle


    std::string cmd00 = "svg(width=6,height=6,pointsize=10,filename=tfile); ";
    std::string cmd01 = "library(qcc);";
    std::string cmd02 = "mu = 100;";
    std::string cmd03 = "sigma_W = 10;";
    std::string cmd04 = "epsilon = rnorm(500);";
    std::string cmd05 = "x = matrix(mu + sigma_W*epsilon, ncol=10, byrow=TRUE);";
    std::string cmd06 = "q = qcc(x, type=\"xbar";
    std::string cmd07 = "\");";
    std::string cmd08 = "dev.off()";
    //std::string cmd08 = "q = qcc(x, type='R');";
    //std::string cmd09 = "q = qcc(x, type='S');";



    cmd = cmd00 + cmd01 + cmd02 + cmd03 + cmd04 + cmd05 + cmd06 + cmd07 + cmd08;
    //cmd = cmd00 + cmd01 + cmd02 + cmd03 + cmd04 + cmd05 + cmd08;

    std::cout << cmd << std::endl;

    m_R.parseEvalQ(cmd);


    //m_R["txt"] = "Hello, world!\n";      // assign a char* (string) to 'txt'
    //m_R.parseEvalQ("cat(txt)");



    filterFile();           	// we need to simplify the svg file for display by Qt

    m_svg->load(m_svgfile);
}

void SPCMainWindow::getBandwidth(int bw) {
    if (bw != m_bw) {
        m_bw = bw;
        plot();
    }
}

void SPCMainWindow::getKernel(int kernel) {
    if (kernel != m_kernel) {
        m_kernel = kernel;
        plot();
    }
}

void SPCMainWindow::getRandomDataCmd(QString txt) {
    m_cmd = txt;
}

void SPCMainWindow::runRandomDataCmd(void) {
    std::string cmd = "y2 <- " + m_cmd.toStdString() + "; y <- y2";
    m_R.parseEvalQNT(cmd);
    plot();                     // after each random draw, update plot with estimate
}

void SPCMainWindow::filterFile() {
    // cairoDevice creates richer SVG than Qt can display
    // but per Michaele Lawrence, a simple trick is to s/symbol/g/ which we do here
    //std::cout << m_tempfile.toStdString() << std::endl;
    QFile infile(m_tempfile);
    infile.open(QFile::ReadOnly);
    QFile outfile(m_svgfile);
    outfile.open(QFile::WriteOnly | QFile::Truncate);



    QTextStream in(&infile);
    QTextStream out(&outfile);
    QRegExp rx1("<symbol");
    QRegExp rx2("</symbol");
    while (!in.atEnd()) {
        QString line = in.readLine();
        //std::cout << line.toStdString() << std::endl;
        line.replace(rx1, "<g"); // so '<symbol' becomes '<g ...'
        line.replace(rx2, "</g");// and '</symbol becomes '</g'
        out << line << "\n";
    }
    infile.close();
    outfile.close();
}
