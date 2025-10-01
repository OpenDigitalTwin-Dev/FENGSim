#include "SPCForm.h"
#include "ui_SPCForm.h"
#include <QFile>
#include <QTextStream>

SPCForm::SPCForm(QWidget *parent) :
    QSvgWidget(parent),
    ui(new Ui::SPCForm)
{
    ui->setupUi(this);
    setupDisplay();
}

SPCForm::~SPCForm()
{
    delete ui;
}

void SPCForm::setupDisplay(void)  {

    this->load(QString("../r/output.svg"));
    //m_svg = new QSvgWidget();
    //setCentralWidget(m_svg);

    //runRandomDataCmd();         // also calls plot()
    //plot();

}

//void SPCForm::plot(void) {




//    const char *kernelstrings[] = { "gaussian", "epanechnikov", "rectangular", "triangular", "cosine" };
//    m_R["bw"] = m_bw;
//    m_R["kernel"] = kernelstrings[m_kernel]; // that passes the string to R
//    std::string cmd0 = "svg(width=6,height=6,pointsize=10,filename=tfile); ";
//    std::string cmd1 = "plot(density(y, bw=bw/100, kernel=kernel), xlim=range(y)+c(-2,2), main=\"Kernel: ";
//    std::string cmd2 = "\"); points(y, rep(0, length(y)), pch=16, col=rgb(0,0,0,1/4));  dev.off()";
//    std::string cmd = cmd0 + cmd1 + kernelstrings[m_kernel] + cmd2; // stick the selected kernel in the middle

//    std::cout << cmd << std::endl;

//    std::string cmd00 = "";
//    std::string cmd01 = "tfile <- \"output.svg\";";
//    std::string cmd02 = "svg(width=6,height=6,pointsize=10,filename=tfile); ";
//    std::string cmd03 = "library(qcc,lib.loc='/home/jiping/FENGSim/toolkit/PS/install/r_install/lib/R/library');";
//    std::string cmd04 = "mu = 100;";
//    std::string cmd05 = "sigma_W = 10;";
//    std::string cmd06 = "epsilon = rnorm(500);";
//    std::string cmd07 = "x = matrix(mu + sigma_W*epsilon, ncol=10, byrow=TRUE);";
//    std::string cmd08 = "q = qcc(x, type=\"xbar";
//    std::string cmd09 = "\");";
//    std::string cmd10 = "dev.off()";
//    //std::string cmd08 = "q = qcc(x, type='R');";
//    //std::string cmd09 = "q = qcc(x, type='S');";

//    cmd = cmd00 + cmd01 + cmd02 + cmd03 + cmd04 + cmd05 + cmd06 + cmd07 + cmd08 + cmd09 + cmd10;
//    std::cout << cmd << std::endl;
//    //cmd = cmd00 + cmd01 + cmd02 + cmd03 + cmd04 + cmd05 + cmd08;


//    m_R.parseEvalQ(cmd);


//    //m_R["txt"] = "Hello, world!\n";      // assign a char* (string) to 'txt'
//    //m_R.parseEvalQ("cat(txt)");

//    return;

//    //filterFile();           	// we need to simplify the svg file for display by Qt


//}

//void SPCForm::getBandwidth(int bw) {
//    if (bw != m_bw) {
//        m_bw = bw;
//        plot();
//    }
//}

//void SPCForm::getKernel(int kernel) {
//    if (kernel != m_kernel) {
//        m_kernel = kernel;
//        plot();
//    }
//}

//void SPCForm::getRandomDataCmd(QString txt) {
//    m_cmd = txt;
//}

//void SPCForm::runRandomDataCmd(void) {
//    std::string cmd = "y2 <- " + m_cmd.toStdString() + "; y <- y2";
//    m_R.parseEvalQNT(cmd);
//    plot();                     // after each random draw, update plot with estimate
//}

//void SPCForm::filterFile() {
//    // cairoDevice creates richer SVG than Qt can display
//    // but per Michaele Lawrence, a simple trick is to s/symbol/g/ which we do here
//    std::cout << m_tempfile.toStdString() << std::endl;
//    std::cout << m_svgfile.toStdString() << std::endl;

//    QFile infile(m_tempfile);
//    infile.open(QFile::ReadOnly);
//    QFile outfile(m_svgfile);
//    outfile.open(QFile::WriteOnly | QFile::Truncate);



//    QTextStream in(&infile);
//    QTextStream out(&outfile);
//    QRegExp rx1("<symbol");
//    QRegExp rx2("</symbol");
//    while (!in.atEnd()) {
//        QString line = in.readLine();
//        //std::cout << line.toStdString() << std::endl;
//        line.replace(rx1, "<g"); // so '<symbol' becomes '<g ...'
//        line.replace(rx2, "</g");// and '</symbol becomes '</g'
//        out << line << "\n";
//    }
//    infile.close();
//    outfile.close();
//}
