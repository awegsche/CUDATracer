/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Mandelbrot sample
    submitted by Mark Granger, NewTek

    CUDA 2.0 SDK - updated with double precision support
    CUDA 2.1 SDK - updated to demonstrate software block scheduling using atomics
    CUDA 2.2 SDK - updated with drawing of Julia sets by Konstantin Kolchin, NVIDIA
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include "Mandelbrot_kernel.h"
#include "Mandelbrot_gold.h"
#include "Testing/test_setup.h"

#include "MinecraftWorld/MCWorld.h"
#include "NBT/nbtfilereader.h"
#include "RenderObjects/Camera.h"

MCWorld* world;
Camera* camera;

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

//Source image on the host side
uchar4 *h_Src = 0;

// Destination image on the GPU side
uchar4 *d_dst = NULL;
float4 *film = NULL;

//Original image width and height
int imageW = 32* 10, imageH = 32 * 8;

#define RENDER_BEAUTIFUL 1
#define RESET_AFTER4 2

int render_mode = 0;
int sample_count = 0;

// Timer ID
StopWatchInterface *hTimer = NULL;

unsigned int pass = 0;

// User interface variables
int lastx = 0;
int lasty = 0;
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

bool haveDoubles = true;
int numSMs = 0;          // number of multiprocessors
int version = 1;             // Compute Capability

// Auto-Verification Code
const int frameCheckNumber = 60;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 15;       // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
float last_time = 0.0f;

int *pArgc = NULL;
char **pArgv = NULL;

const char *sSDKsample = "CUDA Mandelbrot/Julia Set";

#define MAX_EPSILON 50
#define REFRESH_DELAY     4 //ms


#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif
#define BUFFER_DATA(i) ((char *)0 + i)

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// This is specifically to enable the application to enable/disable vsync
typedef BOOL (WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);

void setVSync(int interval)
{
    if (WGL_EXT_swap_control)
    {
        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
        wglSwapIntervalEXT(interval);
    }
}
#endif

void reset_image() {
	if (film)
		cudaFree(film);
	cudaMalloc(&film, sizeof(float4) * imageH * imageW);
	cudaMemset(film, 0, sizeof(float4) * imageH * imageW);
	sample_count = 1;

}



// render Mandelbrot image using CUDA or CPU
void renderImage()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));

	switch (render_mode) {
	case 0:
		camera->render(d_dst, imageW, imageH);
		break;
	case RENDER_BEAUTIFUL:
		camera->expose(d_dst, film, imageW, imageH, sample_count++);
		break;
	case RESET_AFTER4:
		if (sample_count > 4)
			reset_image();
		camera->expose(d_dst, film, imageW, imageH, sample_count++);
		break;
	}
	
	
            
	cudaError_t err =	cudaDeviceSynchronize();  
        
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// OpenGL display function
void displayFunc(void)
{
	
	sdkResetTimer(&hTimer);

	renderImage();

    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

    sdkStopTimer(&hTimer);
    glutSwapBuffers();

	
	float elapsed = sdkGetTimerValue(&hTimer) - last_time;

	//if(elapsed > 250.f)
	float ifps = 1000.f / elapsed;
	char fps[256];
	sprintf(fps, "<CUDA %s Set> %3.1f fps %f %d", "hallo", ifps, sdkGetTimerValue(&hTimer), sample_count);
	glutSetWindowTitle(fps);
	last_time = sdkGetTimerValue(&hTimer);

    //computeFPS();
} // displayFunc

void cleanup()
{
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }

    sdkStopTimer(&hTimer);
    sdkDeleteTimer(&hTimer);

    //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
}

void initMenus() ;

void specialFUNC(int k, int, int) {
	switch (k) {
	case GLUT_KEY_UP:
		camera->rotate_up(.02);
		break;
	case GLUT_KEY_DOWN:
		camera->rotate_down(.02);
		break;
	case GLUT_KEY_LEFT:
		camera->rotate_left(.02);
		break;
	case GLUT_KEY_RIGHT:
		camera->rotate_right(.02);
		break;

	}
}


void switch_render_mode() {
	render_mode = (render_mode + 1) % 3;

	if (render_mode) {
		reset_image();
	}
}

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
    int seed;

    switch (k)
    {
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");

            #if defined(__APPLE__) || defined(MACOSX)
            exit(EXIT_SUCCESS);
            #else
            glutDestroyWindow(glutGetWindow());
            return;
            #endif
            break;

		case 'a':
			camera->move_eye_right(10);
			break;
		case 'w':
			camera->move_eye_forward(15);
			break;
		case 's':
			camera->move_eye_backward(15);
			break;
		case 'd':
			camera->move_eye_left(10);
			break;
		case ' ':
			std::cout << "fskfhuksef";
			switch_render_mode();
			break;
		case '+':
			camera->zoom_in();
			break;
		case '-':
			camera->zoom_out();
			break;
		case '1':
			camera->increase_aperture();
			break;
		case '2':
			camera->decrease_aperture();
			break;

    }

} // keyboardFunc

// OpenGL mouse click function
void clickFunc(int button, int state, int x, int y)
{
    if (button == 0)
    {
        leftClicked = !leftClicked;
    }

    if (button == 1)
    {
        middleClicked = !middleClicked;
    }

    if (button == 2)
    {
        rightClicked = !rightClicked;
    }

    int modifiers = glutGetModifiers();

    if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
    {
        leftClicked = 0;
        middleClicked = 1;
    }

    if (state == GLUT_UP)
    {
        leftClicked = 0;
        middleClicked = 0;

		
    }

    lastx = x;
    lasty = y;
   
} // clickFunc

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
   /* double fx = (double)(x - lastx) / 50.0 / (double)(imageW);
    double fy = (double)(lasty - y) / 50.0 / (double)(imageH);

    if (leftClicked)
    {
        xdOff = fx * scale;
        ydOff = fy * scale;
    }
    else
    {
        xdOff = 0.0f;
        ydOff = 0.0f;
    }

    if (middleClicked)
        if (fy > 0.0f)
        {
            dscale = 1.0 - fy;
            dscale = dscale < 1.05 ? dscale : 1.05;
        }
        else
        {
            dscale = 1.0 / (1.0 + fy);
            dscale = dscale > (1.0 / 1.05) ? dscale : (1.0 / 1.05);
        }
    else
    {
        dscale = 1.0;
    }*/
} // motionFunc

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void mainMenu(int i)
{
   /* precisionMode = i;
    pass = 0;*/
}

void initMenus()
{
  /*  glutCreateMenu(mainMenu);

    if (!g_runCPU)
    {
        glutAddMenuEntry("Hardware single precision", 0);

        if (numSMs > 2)
        {
            glutAddMenuEntry("Emulated double-single precision", 1);
        }

        if (haveDoubles)
        {
            glutAddMenuEntry("Hardware double precision", 2);
        }
    }
    else
    {
        glutAddMenuEntry("Software single precision", 0);
        glutAddMenuEntry("Software double precision", 1);
    }

    glutAttachMenu(GLUT_RIGHT_BUTTON);*/
}

// gl_Shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

void initOpenGLBuffers(int w, int h)
{
    // delete old buffers
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }

    if (gl_Tex)
    {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO)
    {
        //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // allocate new buffers
    h_Src = (uchar4 *)malloc(w * h * 4);

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
    //While a PBO is registered to CUDA, it can't be used
    //as the destination for OpenGL drawing calls.
    //But in our particular case OpenGL is only used
    //to display the content of the PBO, specified by CUDA kernels,
    //so we need to register/unregister it only once.

    // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
                                                 cudaGraphicsMapFlagsWriteDiscard));
    printf("PBO created.\n");

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

	reset_image();
}

void wheelFunc(int wheel, int direction, int x, int y) {
	if (direction > 0)
		camera->increase_d();
	if (direction < 0)
		camera->decrease_d();
}



void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    if (w!=0 && h!=0)  // Do not call when window is minimized that is when width && height == 0
        initOpenGLBuffers(w, h);

    imageW = w;
    imageH = h;
    pass = 0;

    glutPostRedisplay();
}

void initGL(int *argc, char **argv)
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);

    glutDisplayFunc(displayFunc);
    glutSpecialFunc(specialFUNC);
	glutKeyboardFunc(keyboardFunc);
	glutMouseFunc(clickFunc);
	glutMouseWheelFunc(wheelFunc);
	glutMotionFunc(motionFunc);
    glutReshapeFunc(reshapeFunc);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    initMenus();

    if (!isGLVersionSupported(1,5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_SUCCESS);
    }

    printf("OpenGL window created.\n");
}

void initData(int argc, char **argv)
{
    // check for hardware double precision support
    int dev = 0;
    dev = findCudaDevice(argc, (const char **)argv);



    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    version = deviceProp.major*10 + deviceProp.minor;

    numSMs = deviceProp.multiProcessorCount;

    //// initialize some of the arguments
    //if (checkCmdLineFlag(argc, (const char **)argv, "xOff"))
    //{
    //    xOff = getCmdLineArgumentFloat(argc, (const char **)argv, "xOff");
    //}

    //if (checkCmdLineFlag(argc, (const char **)argv, "yOff"))
    //{
    //    yOff = getCmdLineArgumentFloat(argc, (const char **)argv, "yOff");
    //}

    //if (checkCmdLineFlag(argc, (const char **)argv, "scale"))
    //{
    //    scale = getCmdLineArgumentFloat(argc, (const char **)argv, "xOff");
    //}

    //colors.w = 0;
    //colors.x = 3;
    //colors.y = 5;
    //colors.z = 7;
    printf("Data initialization done.\n");
}

//////////////////////////////////////////////////////////////////////////////////
//// runAutoTest validates the Mandelbrot and Julia sets without using OpenGL
//////////////////////////////////////////////////////////////////////////////////
//int runSingleTest(int argc, char **argv)
//{
//    char dump_file[256], *ref_file = NULL;
//    bool haveDouble = false;
//
//    printf("* Running Automatic Test: <%s>\n", sSDKsample);
//
//    strcpy(dump_file, (const char *)"rendered_image.ppm");
//    // We've already determined that file has been passed in as input, we can grab the file here
//    getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
//
//    if (checkCmdLineFlag(argc, (const char **)argv, "fp64"))
//    {
//        haveDouble = true;
//    }
//
//    // initialize Data for CUDA
//    initData(argc, argv);
//
//    // Allocate memory for renderImage (to be able to render into a CUDA memory buffer)
//    checkCudaErrors(cudaMalloc((void **)&d_dst, (imageW * imageH * sizeof(uchar4))));
//
//    //Allocate memory for cpu buffer
//    unsigned char *h_dst =(unsigned char *)malloc(sizeof(uchar4)*imageH*imageW);
//
//    if (g_isJuliaSet)
//    {
//        char *ref_path = sdkFindFilePath("params.txt", argv[0]);
//        startJulia(ref_path);
//
//        for (int i=0; i < 50; i++)
//        {
//            renderImage(false, haveDouble, 0);
//        }
//
//        checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW*imageH*sizeof(uchar4), cudaMemcpyDeviceToHost));
//        sdkSavePPM4ub(dump_file, h_dst, imageW, imageH);
//    }
//    else
//    {
//        // Mandelbrot Set
//        for (int i=0; i < 50; i++)
//        {
//            renderImage(false, haveDouble, 0);
//        }
//
//        checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW*imageH*sizeof(uchar4), cudaMemcpyDeviceToHost));
//        sdkSavePPM4ub(dump_file, h_dst, imageW, imageH);
//    }
//
//    printf("\n[%s], %s Set, %s -> Saved File\n",
//           dump_file,
//           (g_isJuliaSet ? "Julia" : "Mandelbrot"),
//           (haveDouble ? "(fp64 double precision)" : "(fp32 single precision)")
//          );
//
//    if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, argv[0]), MAX_EPSILON_ERROR, 0.15f, false))
//    {
//        printf("Images \"%s\", \"%s\" are different\n", ref_file, dump_file);
//        g_TotalErrors++;
//    }
//    else
//    {
//        printf("Images \"%s\", \"%s\" are matching\n", ref_file, dump_file);
//    }
//
//    checkCudaErrors(cudaFree(d_dst));
//    free(h_dst);
//
//    return true;
//}

//Performance Test
void runBenchmark(int argc, char **argv)
{
    int N = 1000;
    // initialize Data for CUDA
    initData(argc, argv);

    printf("\n* Run Performance Test\n");
    printf("Image Size %d x %d\n",imageW, imageH);
    printf("Double Precision\n");
    printf("%d Iterations\n",N);

    // Allocate memory for renderImage (to be able to render into a CUDA memory buffer)
    checkCudaErrors(cudaMalloc((void **)&d_dst, (imageW * imageH * sizeof(uchar4))));

    float xs, ys;

    // Get the anti-alias sub-pixel sample location
   // GetSample(0, xs, ys);

    double s = 1.0 / (float)imageW;
    double x = (xs - (double)imageW * 0.5f) * s ;
    double y = (ys - (double)imageH * 0.5f) * s ;

    // Create Timers
    StopWatchInterface *kernel_timer = NULL;
    sdkCreateTimer(&kernel_timer);
    sdkStartTimer(&kernel_timer);

    // render Mandelbrot set and verify
    for (int i=0; i < N; i++)
    {
        //RunMandelbrot0(d_dst, imageW, imageH, crunch, x, y,
        //               xJParam, yJParam, s, colors, pass++, animationFrame, 2, numSMs, g_isJuliaSet, version);
        //cudaDeviceSynchronize();
    }

    sdkStopTimer(&hTimer);
    float ExecutionTime = sdkGetTimerValue(&kernel_timer);

    float PixelsPerSecond = (float)imageW*(float)imageH*N/(ExecutionTime/1000.0f);

    printf("\nMegaPixels Per Second %.4f\n",PixelsPerSecond/1e6);

    checkCudaErrors(cudaFree(d_dst));
    sdkDeleteTimer(&kernel_timer);
}

// General initialization call for CUDA Device
void chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
{
    if (bUseOpenGL)
    {
        findCudaGLDevice(argc, argv);
    }
    else
    {
        findCudaDevice(argc, argv);
    }
}

void printHelp()
{
    printf("[Mandelbrot]\n");
    printf("\tUsage Parameters\n");
    printf("\t-device=n        (requires to be in non-graphics mode)\n");
    printf("\t-file=output.ppm (output file for image testing)\n");
    printf("\t-mode=0,1        (0=Mandelbrot Set, 1=Julia Set)\n");
    printf("\t-fp64            (run in double precision mode)\n");
}

void initWorld() {




	world = new MCWorld();

	QString folder = "C:/Users/Andreas.DESKTOP-D87O57E/AppData/Roaming/.minecraft/saves/Alkas/region";

	NBTFileReader* reader = new NBTFileReader(folder, 1, 0);
	reader->Load(world);

#ifndef _DEBUG
	reader = new NBTFileReader(folder, 0, 0);
	reader->Load(world);
	reader = new NBTFileReader(folder, 0, 1);
	reader->Load(world);
#endif

	camera = new Camera();
	camera->set_eye(1000, 200, 1000);
	camera->set_lookat(512, 0, 256);
	
	camera->compute_uvw();
	camera->set_world(world->get_device_world());


}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("[%s] - Starting...\n", sSDKsample);

    // parse command line arguments
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printHelp();
        exit(EXIT_SUCCESS);
    }

    int mode = 0;

   

    //if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    //{
    //    fpsLimit = frameCheckNumber;

    //    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //    findCudaDevice(argc, (const char **)argv); // no OpenGL usage

    //    // We run the Automated Testing code path
    //    runSingleTest(argc, argv);

    //    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    //}
    //else if (checkCmdLineFlag(argc, (const char **)argv, "benchmark"))
    //{
    //    //run benchmark
    //    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //    chooseCudaDevice(argc, (const char **)argv, false); // no OpenGL usage

    //    // We run the Automated Performance Test
    //    runBenchmark(argc, argv);

    //    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    //}
    //// use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //else if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    //{
    //    printf("[%s]\n", argv[0]);
    //    printf("   Does not explicitly support -device=n in OpenGL mode\n");
    //    printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
    //    printf(" > %s -device=n -file=<image_name>.ppm\n", argv[0]);
    //    printf("exiting...\n");
    //    exit(EXIT_SUCCESS);
    //}

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    chooseCudaDevice(argc, (const char **)argv, true); // yes to OpenGL usage

    // Otherwise it succeeds, we will continue to run this sample

    // Initialize OpenGL context first before the CUDA context is created.  This is needed
    // to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);
    initOpenGLBuffers(imageW, imageH);
    initData(argc, argv);
	initWorld();


    //printf("Starting GLUT main loop...\n");
    //printf("\n");
    //printf("Press [s] to toggle between GPU and CPU implementations\n") ;
    //printf("Press [j] to toggle between Julia and Mandelbrot sets\n") ;
    //printf("Press [r] or [R] to decrease or increase red color channel\n") ;
    //printf("Press [g] or [G] to decrease or increase green color channel\n") ;
    //printf("Press [b] or [B] to decrease or increase blue color channel\n") ;
    //printf("Press [e] to reset\n");
    //printf("Press [a] or [A] to animate colors\n");
    //printf("Press [c] or [C] to change colors\n");
    //printf("Press [d] or [D] to increase or decrease the detail\n");
    //printf("Press [p] to record main parameters to file params.txt\n") ;
    //printf("Press [o] to read main parameters from file params.txt\n") ;
    //printf("Left mouse button + drag = move (Mandelbrot or Julia) or animate (Julia)\n");
    //printf("Press [m] to toggle between move and animate (Julia) for left mouse button\n") ;
    //printf("Middle mouse button + drag = Zoom\n");
    //printf("Right mouse button = Menu\n");
    //printf("Press [?] to print location and scale\n");
    //printf("Press [q] to exit\n");
    //printf("\n");

    sdkCreateTimer(&hTimer);
    sdkStartTimer(&hTimer);

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    setVSync(0) ;
#endif

    glutMainLoop();
} // main
