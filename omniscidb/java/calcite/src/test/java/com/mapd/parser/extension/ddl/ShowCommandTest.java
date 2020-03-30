package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;
import com.mapd.thrift.calciteserver.InvalidParseRequest;
import com.mapd.thrift.calciteserver.TPlanResult;

import org.junit.Test;

public class ShowCommandTest extends DDLTest {
  public ShowCommandTest() {
    resourceDirPath = ShowCommandTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "showcommands";
  }

  @Test
  public void showActiveUsers() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_active_users.json");
    final TPlanResult result = processDdlCommand("SHOW ACTIVE USERS;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}
