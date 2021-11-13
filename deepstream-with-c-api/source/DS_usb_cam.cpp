#include "DS_usb_cam.hpp"

static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data){}

static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int deepstream_usb_cam (int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL;
    GstElement * source = NULL, *caps_v4l2src = NULL, *vidconvsrc = NULL, *nvvidconvsrc = NULL, *caps_vidconvsrc = NULL;
    GstElement *streammux = NULL;
    GstElement *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL, *sink = NULL;

    GstCaps *filtercaps = NULL;
    GstElement *transform = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    if (argc != 2){
        g_printerr("Usage %s <USB cam>\n", argv[0]);
        return -1;
    }
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    pipeline = gst_pipeline_new("ds-usbcam-pipeline");

    source = gst_element_factory_make("v4l2src", "usb-cam-source");
    if (!source){
        g_printerr("Unable to create Source \n");
        return -1;
    }

    caps_v4l2src = gst_element_factory_make("capsfilter", "v4l2src-caps");
    if (!caps_v4l2src){
        g_printerr("Unable to create v4l2src capsfilter \n");
        return -1;
    }

    vidconvsrc = gst_element_factory_make("videoconvert", "convertor_src1");
    if (!vidconvsrc){
        g_printerr("Unable to create videoconvert \n");
        return -1;
    }

    nvvidconvsrc = gst_element_factory_make("nvvideoconvert","convertor_src2");
    if (!nvvidconvsrc){
        g_printerr("Unable to create nvvideoconvert");
        return -1;
    }

    caps_vidconvsrc = gst_element_factory_make("capsfilter", "nvmm_caps");
    if (!caps_vidconvsrc){
        g_printerr("Unable to create capsfilter \n");
        return -1;
    }

    streammux = gst_element_factory_make("nvstreamux", "Stream-muxer");
    if (!streammux){
        g_printerr("Unable to create streamux \n");
        return -1;
    }

    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    if (!pgie){
        g_printerr("Unable to create pgie \n");
        return -1;
    }

    nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    if (!nvvidconv){
        g_printerr("Unable to create nvvideoconvert \n");
        return -;
    }

    nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    if (!nvosd){
        g_printerr("Unable to create nvosd \n");
        return -;
    }

    if (prop.integrated){
        transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
    }

    sink = gst_element_factory_make("fakesink", "nvvideo-renderer");
    if (!sink){
        g_printerr("Unable to create fake sink");
        return -1;
    }

    filtercaps = gst_caps_new_simple("video/x-raw", "framerate", GST_TYPE_FRACTION, 30, 1, NULL);

    g_object_set(G_OBJECT(caps_v4l2src),"caps", filtercaps, NULL);
    


}