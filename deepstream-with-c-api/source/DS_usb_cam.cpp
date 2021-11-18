/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "DS_usb_cam.hpp"
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include "gstnvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

#define NVVIDCONV

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
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

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL,
      *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL;
  GstElement *conv1 = NULL, *conv2 = NULL, *nvconv = NULL;
  GstCaps *caps_uyvy = NULL, *caps_nv12_nvmm = NULL, *caps_nv12 = NULL;
  GstCapsFeatures *feature = NULL;
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest1-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("v4l2src", "v4l2-source");
#ifdef NVVIDCONV
  conv1 = gst_element_factory_make ("nvvidconv", "conv1");
  conv2 = gst_element_factory_make ("nvvidconv", "conv2");
#else
  conv1 = gst_element_factory_make ("videoconvert", "conv1");
#endif
  nvconv = gst_element_factory_make ("nvvideoconvert", "nvconv");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!source || !pgie
      || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

#ifdef PLATFORM_TEGRA
  if(!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", 1, "live-source", 1,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "dstest1_pgie_config.txt", NULL);

  g_object_set (G_OBJECT (source), "device", "/dev/video1", "num-buffers", 900, NULL);

  caps_uyvy = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "UYVY",
      "width", G_TYPE_INT, 1920, "height", G_TYPE_INT,
      1080, "framerate", GST_TYPE_FRACTION,
      30, 1, NULL);
  caps_nv12_nvmm = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12", NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps_nv12_nvmm, 0, feature);
  caps_nv12 = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12", NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      source, conv1, nvconv, streammux, pgie,
      nvvidconv, nvosd, transform, sink, NULL);

  gst_element_link_filtered(source, conv1, caps_uyvy);
#ifdef NVVIDCONV
  gst_bin_add (GST_BIN (pipeline), conv2);
  gst_element_link_filtered(conv1, conv2, caps_nv12_nvmm);
  gst_element_link_filtered(conv2, nvconv, caps_nv12);
#else
  gst_element_link_filtered(conv1, nvconv, caps_nv12);
#endif
  gst_element_link_pads_filtered(nvconv, "src", streammux, "sink_0", caps_nv12_nvmm);

  gst_caps_unref(caps_uyvy);
  gst_caps_unref(caps_nv12_nvmm);
  gst_caps_unref(caps_nv12);

  if (!gst_element_link_many (streammux, pgie,
      nvvidconv, nvosd, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}


// #include "DS_usb_cam.hpp"

// static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data){}

// static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data)
// {
//   GMainLoop *loop = (GMainLoop *) data;
//   switch (GST_MESSAGE_TYPE (msg)) {
//     case GST_MESSAGE_EOS:
//       g_print ("End of stream\n");
//       g_main_loop_quit (loop);
//       break;
//     case GST_MESSAGE_ERROR:{
//       gchar *debug;
//       GError *error;
//       gst_message_parse_error (msg, &error, &debug);
//       g_printerr ("ERROR from element %s: %s\n",
//           GST_OBJECT_NAME (msg->src), error->message);
//       if (debug)
//         g_printerr ("Error details: %s\n", debug);
//       g_free (debug);
//       g_error_free (error);
//       g_main_loop_quit (loop);
//       break;
//     }
//     default:
//       break;
//   }
//   return TRUE;
// }

// int deepstream_usb_cam (int argc, char *argv[])
// {
//     GMainLoop *loop = NULL;
//     GstElement *pipeline = NULL;
//     GstElement * source = NULL, *caps_v4l2src = NULL, *vidconvsrc = NULL, *nvvidconvsrc = NULL, *caps_vidconvsrc = NULL;
//     GstElement *streammux = NULL;
//     GstElement *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL, *sink = NULL;

//     GstCaps *filtercaps = NULL;
//     GstElement *transform = NULL;
//     GstBus *bus = NULL;
//     guint bus_watch_id;
//     GstPad *osd_sink_pad = NULL;

//     int current_device = -1;
//     cudaGetDevice(&current_device);
//     struct cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, current_device);

//     if (argc != 2){
//         g_printerr("Usage %s <USB cam>\n", argv[0]);
//         return -1;
//     }
//     gst_init(&argc, &argv);
//     loop = g_main_loop_new(NULL, FALSE);

//     pipeline = gst_pipeline_new("ds-usbcam-pipeline");

//     source = gst_element_factory_make("v4l2src", "usb-cam-source");
//     if (!source){
//         g_printerr("Unable to create Source \n");
//         return -1;
//     }

//     caps_v4l2src = gst_element_factory_make("capsfilter", "v4l2src-caps");
//     if (!caps_v4l2src){
//         g_printerr("Unable to create v4l2src capsfilter \n");
//         return -1;
//     }

//     vidconvsrc = gst_element_factory_make("videoconvert", "convertor_src1");
//     if (!vidconvsrc){
//         g_printerr("Unable to create videoconvert \n");
//         return -1;
//     }

//     nvvidconvsrc = gst_element_factory_make("nvvideoconvert","convertor_src2");
//     if (!nvvidconvsrc){
//         g_printerr("Unable to create nvvideoconvert");
//         return -1;
//     }

//     caps_vidconvsrc = gst_element_factory_make("capsfilter", "nvmm_caps");
//     if (!caps_vidconvsrc){
//         g_printerr("Unable to create capsfilter \n");
//         return -1;
//     }

//     streammux = gst_element_factory_make("nvstreamux", "Stream-muxer");
//     if (!streammux){
//         g_printerr("Unable to create streamux \n");
//         return -1;
//     }

//     pgie = gst_element_factory_make("nvinfer", "primary-inference");
//     if (!pgie){
//         g_printerr("Unable to create pgie \n");
//         return -1;
//     }

//     nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
//     if (!nvvidconv){
//         g_printerr("Unable to create nvvideoconvert \n");
//         return -;
//     }

//     nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
//     if (!nvosd){
//         g_printerr("Unable to create nvosd \n");
//         return -;
//     }

//     if (prop.integrated){
//         transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
//     }

//     sink = gst_element_factory_make("fakesink", "nvvideo-renderer");
//     if (!sink){
//         g_printerr("Unable to create fake sink");
//         return -1;
//     }

//     filtercaps = gst_caps_new_simple("video/x-raw", "framerate", GST_TYPE_FRACTION, 30, 1, NULL);

//     g_object_set(G_OBJECT(caps_v4l2src),"caps", filtercaps, NULL);
    
// }